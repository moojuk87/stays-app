from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bson import ObjectId
import aiohttp
import httpx
from icalendar import Calendar
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from typing import Optional
import os
import json
import base64

# ── 환경변수
MONGO_URL          = os.environ.get("MONGO_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ── DB
client = AsyncIOMotorClient(MONGO_URL)
db = client.stays_db


# ─────────────────────────────────────────
#  텔레그램 유틸
# ─────────────────────────────────────────
async def send_telegram(text: str, reply_markup=None):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with httpx.AsyncClient() as c:
        await c.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)


# ─────────────────────────────────────────
#  Gemini 파싱
# ─────────────────────────────────────────
async def parse_with_gemini(text: str = None, image_bytes: bytes = None, image_mime: str = None) -> list:
    kst = timezone(timedelta(hours=9))
    today_str = datetime.now(kst).strftime("%Y-%m-%d")

    system_prompt = f"""오늘 날짜: {today_str} (KST)

아래 텍스트(또는 이미지)에서 일정이나 할일을 모두 추출하세요.
상대적 날짜("이번주 토요일", "내일" 등)는 오늘 기준 절대 날짜로 변환하세요.

반드시 JSON 배열만 반환하세요. 마크다운 펜스 없이 순수 JSON만.
형식: [{{"title": "일정 제목", "datetime": "YYYY-MM-DDTHH:MM", "memo": "원문 요약"}}]

datetime이 불명확하면 null로 설정하세요.
일정이 없으면 빈 배열 []을 반환하세요."""

    parts = [{"text": system_prompt}]
    if image_bytes:
        parts.append({"inline_data": {
            "mime_type": image_mime or "image/jpeg",
            "data": base64.b64encode(image_bytes).decode()
        }})
    if text:
        parts.append({"text": f"\n추출 대상:\n{text}"})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient() as c:
        resp = await c.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────
#  리마인더 스케줄러
# ─────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone="Asia/Seoul")

REMINDER_LABELS = {
    "7d": "📅 7일 전", "5d": "📅 5일 전", "3d": "⏰ 3일 전",
    "1d": "🔔 내일",   "3h": "🚨 3시간 전", "same_day": "🌅 오늘",
}

async def check_reminders():
    try:
        kst = timezone(timedelta(hours=9))
        now = datetime.now(kst).replace(tzinfo=None)
        schedules = await db.schedules.find({"done": False}).to_list(300)

        for sched in schedules:
            dt_str = sched.get("datetime")
            if not dt_str:
                continue
            try:
                sched_dt = datetime.fromisoformat(dt_str)
            except Exception:
                continue

            diff_min = (sched_dt - now).total_seconds() / 60
            notified = sched.get("notified", [])

            checks = [
                ("7d", 7*24*60,   6.5*24*60),
                ("5d", 5*24*60,   4.5*24*60),
                ("3d", 3*24*60,   2.5*24*60),
                ("1d", 1*24*60,   0.5*24*60),
                ("3h", 3*60,       2.5*60),
            ]
            for key, upper, lower in checks:
                if key not in notified and lower <= diff_min <= upper:
                    msg = (f"{REMINDER_LABELS[key]} 알림\n\n"
                           f"<b>{sched['title']}</b>\n"
                           f"📅 {sched_dt.strftime('%m월 %d일 %H:%M')}")
                    if sched.get("memo"):
                        msg += f"\n📝 {sched['memo']}"
                    markup = {"inline_keyboard": [[
                        {"text": "✅ 완료 처리", "callback_data": f"done_{str(sched['_id'])}"}
                    ]]}
                    await send_telegram(msg, markup)
                    await db.schedules.update_one(
                        {"_id": sched["_id"]}, {"$push": {"notified": key}})

            # 당일 아침 9시 이후
            if "same_day" not in notified and sched_dt.date() == now.date() and now.hour >= 9:
                msg = (f"🌅 오늘 일정 알림\n\n"
                       f"<b>{sched['title']}</b>\n"
                       f"📅 오늘 {sched_dt.strftime('%H:%M')}")
                markup = {"inline_keyboard": [[
                    {"text": "✅ 완료 처리", "callback_data": f"done_{str(sched['_id'])}"}
                ]]}
                await send_telegram(msg, markup)
                await db.schedules.update_one(
                    {"_id": sched["_id"]}, {"$push": {"notified": "same_day"}})

    except Exception as e:
        print(f"[리마인더 오류] {e}")


# ─────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(check_reminders, "interval", minutes=10, id="reminders")
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  Models
# ─────────────────────────────────────────
class Property(BaseModel):
    property_id: str
    name: str
    ical_url: str
    bookings: list = []
    last_synced: Optional[str] = None

class ScheduleCreate(BaseModel):
    title: str
    datetime: Optional[str] = None
    memo: Optional[str] = None

def sched_to_dict(s: dict) -> dict:
    s["id"] = str(s.pop("_id"))
    return s


# ═══════════════════════════════════════════
#  기존 Properties API — 변경 없음
# ═══════════════════════════════════════════

@app.get("/api/properties")
async def get_properties():
    props = await db.properties.find({}, {"_id": 0}).to_list(100)
    return props

@app.post("/api/properties")
async def create_property(prop: Property):
    existing = await db.properties.find_one({"property_id": prop.property_id})
    if existing:
        return prop
    await db.properties.insert_one(prop.dict())
    return prop

@app.delete("/api/properties/{id}")
async def delete_property(id: str):
    result = await db.properties.delete_one({"property_id": id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
    return {"status": "deleted"}

@app.post("/api/properties/{id}/sync")
async def sync_ical(id: str):
    prop = await db.properties.find_one({"property_id": id})
    if not prop:
        raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(prop["ical_url"], timeout=aiohttp.ClientTimeout(total=10)) as resp:
                ics_data = await resp.text()
        cal = Calendar.from_ical(ics_data)
        bookings = []
        today = datetime.utcnow().date()
        for c in cal.walk("VEVENT"):
            status = str(c.get("status", "")).upper()
            if status == "CANCELLED":
                continue
            summary = str(c.get("summary", "")).strip()
            if any(kw in summary.lower() for kw in ["not available", "unavailable", "blocked", "준비중", "청소"]):
                continue
            start = c.get("dtstart")
            end = c.get("dtend")
            if start and end:
                end_dt = end.dt if hasattr(end.dt, 'year') else end.dt
                end_date = end_dt.date() if hasattr(end_dt, 'date') else end_dt
                if end_date < today:
                    continue
                bookings.append({
                    "start": str(start.dt),
                    "end": str(end.dt),
                    "summary": summary
                })
        now = datetime.utcnow().isoformat()
        await db.properties.update_one(
            {"property_id": id},
            {"$set": {"bookings": bookings, "last_synced": now}}
        )
        return {"status": "success", "bookings": bookings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동기화 실패: {str(e)}")

@app.get("/api/properties/search")
async def search_properties(start: datetime, end: datetime):
    props = await db.properties.find({}, {"_id": 0}).to_list(100)
    available = []
    for p in props:
        bookings = p.get("bookings", [])
        conflict = any(
            datetime.fromisoformat(str(b["start"])[:19]) < end and
            datetime.fromisoformat(str(b["end"])[:19]) > start
            for b in bookings
        )
        if not conflict:
            available.append(p)
    return available


# ═══════════════════════════════════════════
#  Schedules API (신규)
# ═══════════════════════════════════════════

@app.get("/api/schedules")
async def get_schedules():
    scheds = await db.schedules.find().sort("datetime", 1).to_list(500)
    return [sched_to_dict(s) for s in scheds]

@app.post("/api/schedules")
async def create_schedule(sched: ScheduleCreate):
    doc = {
        "title":      sched.title,
        "datetime":   sched.datetime,
        "memo":       sched.memo or "",
        "done":       False,
        "notified":   [],
        "created_at": datetime.utcnow().isoformat()
    }
    result = await db.schedules.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    return doc

@app.put("/api/schedules/{id}/done")
async def mark_done(id: str):
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")
    result = await db.schedules.update_one({"_id": oid}, {"$set": {"done": True}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다.")
    return {"status": "done"}

@app.put("/api/schedules/{id}/undone")
async def mark_undone(id: str):
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")
    await db.schedules.update_one({"_id": oid}, {"$set": {"done": False}})
    return {"status": "undone"}

@app.delete("/api/schedules/{id}")
async def delete_schedule(id: str):
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")
    result = await db.schedules.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다.")
    return {"status": "deleted"}


# ═══════════════════════════════════════════
#  Telegram Webhook
# ═══════════════════════════════════════════

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()

    # 인라인 버튼 콜백 처리
    if "callback_query" in data:
        cq      = data["callback_query"]
        cq_data = cq.get("data", "")
        cq_id   = cq["id"]
        if cq_data.startswith("done_"):
            sched_id = cq_data[5:]
            try:
                oid   = ObjectId(sched_id)
                sched = await db.schedules.find_one({"_id": oid})
                if sched:
                    await db.schedules.update_one({"_id": oid}, {"$set": {"done": True}})
                    async with httpx.AsyncClient() as c:
                        await c.post(f"{TELEGRAM_API}/answerCallbackQuery",
                                     json={"callback_query_id": cq_id, "text": "✅ 완료 처리됐습니다!"})
                    await send_telegram(f"✅ <b>{sched['title']}</b> 완료 처리됐습니다.")
            except Exception as e:
                print(f"[콜백 오류] {e}")
        return {"ok": True}

    # 일반 메시지
    msg     = data.get("message", {})
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if chat_id != str(TELEGRAM_CHAT_ID):
        return {"ok": True}

    text  = msg.get("text", "").strip()
    photo = msg.get("photo")

    if text == "/list":
        scheds = await db.schedules.find({"done": False}).sort("datetime", 1).to_list(20)
        if not scheds:
            await send_telegram("📋 등록된 일정이 없습니다.")
        else:
            lines = ["📋 <b>현재 일정 목록</b>\n"]
            for s in scheds:
                dt_label = ""
                if s.get("datetime"):
                    try:
                        dt_label = datetime.fromisoformat(s["datetime"]).strftime("%m/%d %H:%M ")
                    except Exception:
                        pass
                lines.append(f"• {dt_label}<b>{s['title']}</b>")
            await send_telegram("\n".join(lines))
        return {"ok": True}

    if text == "/today":
        kst   = timezone(timedelta(hours=9))
        today = datetime.now(kst).date()
        scheds = await db.schedules.find({"done": False}).to_list(300)
        today_list = [s for s in scheds if s.get("datetime") and
                      _safe_date(s["datetime"]) == today]
        if not today_list:
            await send_telegram("✨ 오늘 일정이 없습니다.")
        else:
            lines = [f"📅 <b>오늘({today.strftime('%m/%d')}) 일정</b>\n"]
            for s in today_list:
                dt_label = ""
                try:
                    dt_label = datetime.fromisoformat(s["datetime"]).strftime("%H:%M ")
                except Exception:
                    pass
                lines.append(f"• {dt_label}<b>{s['title']}</b>")
            await send_telegram("\n".join(lines))
        return {"ok": True}

    if not text and not photo:
        return {"ok": True}

    await send_telegram("⏳ 일정을 분석하는 중...")

    try:
        if photo:
            file_id = photo[-1]["file_id"]
            async with httpx.AsyncClient() as c:
                f_resp   = await c.get(f"{TELEGRAM_API}/getFile?file_id={file_id}")
                f_path   = f_resp.json()["result"]["file_path"]
                img_resp = await c.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{f_path}")
                image_bytes = img_resp.content
            parsed = await parse_with_gemini(image_bytes=image_bytes)
        else:
            parsed = await parse_with_gemini(text=text)

        if not parsed:
            await send_telegram("❌ 일정을 찾을 수 없었습니다.\n일정이 포함된 텍스트나 캡처 이미지를 보내주세요.")
            return {"ok": True}

        saved = []
        for item in parsed:
            doc = {
                "title":      item.get("title", "제목 없음"),
                "datetime":   item.get("datetime"),
                "memo":       item.get("memo", ""),
                "done":       False,
                "notified":   [],
                "created_at": datetime.utcnow().isoformat()
            }
            result = await db.schedules.insert_one(doc)
            saved.append({"id": str(result.inserted_id), **doc})

        lines = [f"✅ <b>{len(saved)}개 일정이 등록됐습니다!</b>\n"]
        inline_buttons = []
        for s in saved:
            dt_label = ""
            if s.get("datetime"):
                try:
                    dt_label = f"\n   📅 {datetime.fromisoformat(s['datetime']).strftime('%m월 %d일 %H:%M')}"
                except Exception:
                    pass
            lines.append(f"• <b>{s['title']}</b>{dt_label}")
            inline_buttons.append([
                {"text": f"✅ {s['title'][:20]} 완료", "callback_data": f"done_{s['id']}"}
            ])

        await send_telegram("\n".join(lines), {"inline_keyboard": inline_buttons})

    except Exception as e:
        print(f"[파싱 오류] {e}")
        await send_telegram(f"❌ 파싱 중 오류가 발생했습니다: {str(e)[:100]}")

    return {"ok": True}


def _safe_date(dt_str):
    try:
        return datetime.fromisoformat(dt_str).date()
    except Exception:
        return None
