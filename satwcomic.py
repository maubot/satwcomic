# satwcomic - A maubot plugin to view SatWComics
# Copyright (C) 2020 Tulir Asokan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Optional, Type, List, Iterable, Tuple, Union
from difflib import SequenceMatcher
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import asyncio
import random
import re
import os

from sqlalchemy import Column, String, Integer, Text, BigInteger, orm
from sqlalchemy.ext.declarative import declarative_base
from pyquery import PyQuery
from lxml.html import HtmlElement
from attr import dataclass
from yarl import URL

from mautrix.types import (ContentURI, RoomID, UserID, ImageInfo, SerializableAttrs, EventType,
                           StateEvent)
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper
from maubot import Plugin, MessageEvent
from maubot.handlers import command, event

try:
    import magic
except ImportError:
    magic = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class SatWInfo(SerializableAttrs):
    slug: str
    image_url: str
    date: datetime
    title: str
    body: str
    countries: List[str]


class MediaCache:
    __tablename__ = "media_cache"
    query: orm.Query = None

    url: str = Column(String(255), primary_key=True)
    mxc_uri: ContentURI = Column(String(255))
    mime_type: str = Column(String(255))
    filename: str = Column(String(255))
    width: int = Column(Integer)
    height: int = Column(Integer)
    size: int = Column(Integer)

    def __init__(self, url: str, mxc_uri: ContentURI, mime_type: str, filename: str,
                 width: int, height: int, size: int) -> None:
        self.url = url
        self.mxc_uri = mxc_uri
        self.mime_type = mime_type
        self.filename = filename
        self.width = width
        self.height = height
        self.size = size


class SatWIndex:
    __tablename__ = "satw_index"
    query: orm.Query = None

    slug: str = Column(String(255), primary_key=True)
    image_url: str = Column(String(255))
    timestamp: int = Column(BigInteger)
    title: str = Column(Text)
    body: str = Column(Text)
    countries: str = Column(Text)

    def __init__(self, slug: str, image_url: str, timestamp: Union[int, datetime],
                 title: str, body: str, countries: Union[str, List[str]]) -> None:
        self.slug = slug
        self.image_url = image_url
        self.timestamp = timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp
        self.title = title
        self.body = body
        self.countries = ",".join(countries) if isinstance(countries, list) else countries

    def country_list(self) -> List[str]:
        return self.countries.split(",")

    def date(self) -> datetime:
        return datetime.utcfromtimestamp(self.timestamp)

    def __lt__(self, other: 'SatWIndex') -> bool:
        return self.timestamp > other.timestamp

    def __gt__(self, other: 'SatWIndex') -> bool:
        return self.timestamp < other.timestamp

    def __eq__(self, other: 'SatWIndex') -> bool:
        return self.timestamp == other.timestamp


class Subscriber:
    __tablename__ = "subscriber"
    query: orm.Query = None

    room_id: RoomID = Column(String(255), primary_key=True)
    requested_by: UserID = Column(String(255))

    def __init__(self, room_id: RoomID, requested_by: UserID) -> None:
        self.room_id = room_id
        self.requested_by = requested_by


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("inline")
        helper.copy("poll_interval")
        helper.copy("spam_sleep")
        helper.copy("allow_reindex")
        helper.copy("max_search_results")
        helper.copy("base_command")


class SatWBot(Plugin):
    media_cache: Type[MediaCache]
    subscriber: Type[Subscriber]
    satw_index: Type[SatWIndex]
    db: orm.Session
    latest_slug: str
    poll_task: asyncio.Future

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    async def start(self) -> None:
        await super().start()
        self.config.load_and_update()
        db_factory = orm.sessionmaker(bind=self.database)
        db_session = orm.scoped_session(db_factory)

        base = declarative_base()
        base.metadata.bind = self.database

        class MediaCacheImpl(MediaCache, base):
            query = db_session.query_property()

        class SatWIndexImpl(SatWIndex, base):
            query = db_session.query_property()

        class SubscriberImpl(Subscriber, base):
            query = db_session.query_property()

        self.media_cache = MediaCacheImpl
        self.subscriber = SubscriberImpl
        self.satw_index = SatWIndexImpl
        base.metadata.create_all()

        self.db = db_session
        self.latest_slug = ""

        self.poll_task = asyncio.ensure_future(self.poll_comic(), loop=self.loop)

    async def stop(self) -> None:
        await super().stop()
        self.poll_task.cancel()

    def _index_info(self, info: SatWInfo) -> None:
        self.db.merge(self.satw_index(info.slug, info.image_url, info.date,
                                      info.title, info.body, info.countries))
        self.db.commit()

    def _parse_body(self, elem: HtmlElement) -> str:
        text = []
        if elem.tag == "br":
            text.append("\n")
        elif elem.tag == "img":
            try:
                text.append(elem.attrib["alt"])
            except KeyError:
                pass
        else:
            if elem.text:
                text.append(elem.text.strip("\r\n"))
            for child in elem.iterchildren():
                text.append(self._parse_body(child))
        if elem.tail:
            text.append(elem.tail.strip("\r\n"))
        return "".join(text)

    async def get_comic(self, slug: str) -> Tuple[Optional[SatWInfo], str]:
        resp = await self.http.get(f"https://satwcomic.com/{slug}")
        if resp.status != 200:
            resp.raise_for_status()
            return None, ""
        data = await resp.text(errors="ignore")
        q = PyQuery(data)
        try:
            if slug == "latest":
                url = q("a.btn[title^='Permanent link']").attr("href")
                slug = url[len("https://satwcomic.com/"):]
            image_url = q(".card > center img[itemprop='image']").attr("src").replace("/core", "")
            title = q("[itemprop='headline']").text()
            body = self._parse_body(q("[itemprop='articleBody']")[0])
            countries = [item.text for item
                         in q(".card-body a[href^='https://satwcomic.com/the-world/']")]

            date_str = q(".float-right.text-muted > small")[0].text
            date_str = re.sub(r"^(\d+)(st|nd|rd|th)", r"\1", date_str)
            date = datetime.strptime(date_str, "%d %B %Y").replace(tzinfo=timezone.utc)

            info = SatWInfo(slug=slug, image_url=image_url, date=date,
                            title=title, body=body, countries=countries)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError):
            self.log.warning(f"Failed to parse comic {slug}", exc_info=True)
            return None, ""
        try:
            previous_url = q("a[accesskey='p']").attr("href")
            previous_slug = previous_url[len("https://satwcomic.com/"):]
        except (KeyError, ValueError, TypeError, IndexError, AttributeError):
            previous_slug = ""
        self._index_info(info)
        return info, previous_slug

    async def _get_media_info(self, url: str) -> Optional[MediaCache]:
        cache = self.media_cache.query.get(url)
        if cache is not None:
            return cache

        resp = await self.http.get(url)
        if resp.status != 200:
            self.log.warning(f"Unexpected status fetching image {url}: {resp.status}")
            return None

        data = await resp.read()
        if magic:
            mime = magic.from_buffer(data, mime=True)
        else:
            mime = "image/png" if url.endswith("png") else "image/jpeg"
        if Image is not None:
            img = Image.open(BytesIO(data))
            width, height = img.size
        else:
            width, height = None, None

        filename = os.path.basename(URL(url).path)
        uri = await self.client.upload_media(data, mime_type=mime, filename=filename)
        cache = self.media_cache(url=url, mxc_uri=uri, mime_type=mime, filename=filename,
                                 width=width, height=height, size=len(data))
        self.db.add(cache)
        self.db.commit()
        return cache

    async def send_comic(self, room_id: RoomID, comic: SatWInfo) -> None:
        try:
            await self._send_comic(room_id, comic)
        except Exception:
            self.log.exception(f"Failed to send SatWComic {comic.slug} to {room_id}")

    async def _send_comic(self, room_id: RoomID, comic: SatWInfo) -> None:
        info = await self._get_media_info(comic.image_url)
        body_html = escape(comic.body).replace("\n", "<br>")
        format = self.config["format"]
        if format == "inline":
            await self.client.send_text(room_id,
                                        text=f"{info.url}\n"
                                             f"# {comic.title}\n"
                                             f"{comic.body}",
                                        html=(f"<img src='{info.mxc_uri}' title='{comic.slug}'/>"
                                              f"<h1>{escape(comic.title)}</h1>"
                                              f"<p>{body_html}</p>"))
        elif format in ("separate", "filename"):
            filename = info.filename if format == "separate" else comic.title
            await self.client.send_image(room_id, url=info.mxc_uri, file_name=filename,
                                         info=ImageInfo(
                                             mimetype=info.mime_type,
                                             size=info.size,
                                             width=info.width,
                                             height=info.height,
                                         ))
            if format == "separate":
                await self.client.send_text(room_id, text=f"# {comic.title}\n{comic.body}",
                                            html=f"<h1>{escape(comic.title)}</h1>"
                                                 f"<p>{body_html}</p>")
        else:
            self.log.error(f"Unknown format \"{self.config['format']}\" specified in config.")

    async def broadcast(self, comic: SatWInfo) -> None:
        self.log.debug(f"Broadcasting SatWComic {comic.title}")
        subscribers = list(self.subscriber.query.all())
        random.shuffle(subscribers)
        spam_sleep = self.config["spam_sleep"]
        if spam_sleep < 0:
            await asyncio.gather(*[self.send_comic(sub.room_id, comic)
                                   for sub in subscribers],
                                 loop=self.loop)
        else:
            for sub in subscribers:
                await self.send_comic(sub.room_id, comic)
                if spam_sleep > 0:
                    await asyncio.sleep(spam_sleep)

    async def poll_comic(self) -> None:
        try:
            await self._poll_comic()
        except asyncio.CancelledError:
            self.log.debug("Polling stopped")
            pass
        except Exception:
            self.log.exception("Failed to poll SatWComic")

    async def _poll_comic(self) -> None:
        self.log.debug("Polling started")
        latest, _ = await self.get_comic("latest")
        self.latest_slug = latest.slug
        while True:
            await asyncio.sleep(self.config["poll_interval"], loop=self.loop)
            try:
                latest, _ = await self.get_comic("latest")
                if latest.slug != self.latest_slug:
                    self.latest_slug = latest.slug
                    await self.broadcast(latest)
            except Exception:
                self.log.exception("Failed to poll SatWComic")

    @command.new(name=lambda self: self.config["base_command"],
                 help="Search for a comic and view the first result, or view the latest comic",
                 require_subcommand=False, arg_fallthrough=False)
    @command.argument("query", required=False, pass_raw=True)
    async def comic(self, evt: MessageEvent, query: Optional[str]) -> None:
        if query:
            results = self._search(query)
            if not results:
                if query.islower() and " " not in query:
                    comic, _ = await self.get_comic(query)
                    if not comic:
                        await evt.reply("No results :(")
                        return
                else:
                    await evt.reply("No results :(")
                    return
            else:
                result = results[0][0]
                comic, _ = await self.get_comic(result.slug)
                if not comic:
                    await evt.reply(f"Found result {result.title}, but failed to fetch content")
                    return
        else:
            comic, _ = await self.get_comic("latest")
        await self.send_comic(evt.room_id, comic)

    @comic.subcommand("reindex", help="Fetch and store info about every SatWComic to date for "
                                      "searching.")
    async def reindex(self, evt: MessageEvent) -> None:
        if not self.config["allow_reindex"]:
            await evt.reply("Sorry, the reindex command has been disabled on this instance.")
            return
        self.config["allow_reindex"] = False
        self.config.save()
        await evt.reply("Reindexing comic database...")
        indexed = 0
        prev_slug = "latest"
        while prev_slug:
            comic, prev_slug = await self.get_comic(prev_slug)
            if comic:
                self.log.debug(f"Indexed {comic.title}")
                indexed += 1
            else:
                self.log.warning(f"Failed to fetch {prev_slug}")
        await evt.reply(f"Reindexing complete. Indexed {indexed} comics.")

    def _index_similarity(self, result: SatWIndex, query: str) -> float:
        query = query.lower()
        if result.slug == query:
            return 9001
        slug_sim = SequenceMatcher(None, result.slug.strip().lower(), query).ratio()
        title_sim = SequenceMatcher(None, result.title.strip().lower(), query).ratio()
        body_sim = SequenceMatcher(None, result.body.strip().lower(), query).ratio()
        sim = max(slug_sim, title_sim, body_sim)
        return round(sim * 100, 1)

    def _sort_search_results(self, results: List[SatWIndex], query: str
                             ) -> Iterable[Tuple[SatWIndex, float]]:
        similarity = (self._index_similarity(result, query) for result in results)
        return ((result, similarity) for similarity, result
                in sorted(zip(similarity, results), reverse=True))

    def _search(self, query: str) -> Optional[List[Tuple[SatWIndex, float]]]:
        sql_query = f"%{query}%"
        results = self.satw_index.query.filter(self.satw_index.slug.like(sql_query)
                                               | self.satw_index.title.like(sql_query)
                                               | self.satw_index.body.like(sql_query)).all()
        if len(results) == 0:
            return None
        return list(self._sort_search_results(results, query))

    @comic.subcommand("search", help="Search for a relevant SatWComic")
    @command.argument("query", pass_raw=True)
    async def search(self, evt: MessageEvent, query: str) -> None:
        results = self._search(query)
        if not results:
            await evt.reply("No results :(")
            return
        msg = "Results:\n\n"
        more_results = None
        limit = self.config["max_search_results"]
        if len(results) > limit:
            more_results = len(results) - limit, results[limit][1]
            results = results[:limit]
        msg += "\n".join(f"* [{result.title}](https://satwcomic.com/{result.slug}"
                         f"  ({similarity} % match)"
                         for result, similarity in results)
        if more_results:
            number, similarity = more_results
            msg += (f"\n\nThere were {number} other results "
                    f"with a similarity lower than {similarity + 0.1} %")
        await evt.reply(msg)

    @comic.subcommand("subscribe", help="Subscribe to SatWComic updates")
    async def subscribe(self, evt: MessageEvent) -> None:
        sub = self.subscriber.query.get(evt.room_id)
        if sub is not None:
            await evt.reply("This room has already been subscribed to "
                            f"SatWComic updates by {sub.requested_by}")
            return
        sub = self.subscriber(evt.room_id, evt.sender)
        self.db.add(sub)
        self.db.commit()
        await evt.reply("Subscribed to SatWComic updates successfully!")

    @comic.subcommand("unsubscribe", help="Unsubscribe from SatWComic updates")
    async def unsubscribe(self, evt: MessageEvent) -> None:
        sub = self.subscriber.query.get(evt.room_id)
        if sub is None:
            await evt.reply("This room is not subscribed to SatWComic updates.")
            return
        self.db.delete(sub)
        self.db.commit()
        await evt.reply("Unsubscribed from SatWComic updates successfully :(")

    @event.on(EventType.ROOM_TOMBSTONE)
    async def tombstone(self, evt: StateEvent) -> None:
        if not evt.content.replacement_room:
            return
        sub = self.db.query(self.subscriber).get(evt.room_id)
        if sub:
            sub.room_id = evt.content.replacement_room
            self.db.commit()
