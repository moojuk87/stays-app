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
import calendar as cal

# ── 환경변수
MONGO_URL          = os.environ.get("MONGO_URL")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ── DB
client = AsyncIOMotorClient(MONGO_URL)
db = client.stays_db

# ── Groq 설정
GROQ_API_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TEXT_MODEL     = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL   = "llama-3.2-11b-vision-preview"


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
#  Groq 파싱
# ─────────────────────────────────────────
async def parse_with_groq(text: str = None, image_bytes: bytes = None, image_mime: str = None) -> list:
    kst = timezone(timedelta(hours=9))
    today_str = datetime.now(kst).strftime("%Y-%m-%d")

    system_prompt = f"""오늘 날짜: {today_str} (KST)

아래 텍스트(또는 이미지)에서 일정이나 할일을 모두 추출하세요.
상대적 날짜("이번주 토요일", "내일" 등)는 오늘 기준 절대 날짜로 변환하세요.

반드시 JSON 배열만 반환하세요. 마크다운 펜스 없이 순수 JSON만.
형식: [{{"title": "일정 제목", "datetime": "YYYY-MM-DDTHH:MM", "memo": "원문 요약"}}]

datetime이 불명확하면 null로 설정하세요.
일정이 없으면 빈 배열 []을 반환하세요."""

    if image_bytes:
        model = GROQ_VISION_MODEL
        b64   = base64.b64encode(image_bytes).decode()
        mime  = image_mime or "image/jpeg"
        messages = [{"role": "user", "content": [
            {"type": "text",      "text": system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        ]}]
    else:
        model    = GROQ_TEXT_MODEL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"추출 대상:\n{text}"}
        ]

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.1,
        "max_tokens":  1000,
    }

    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                GROQ_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
            # 단일 객체로 왔을 경우 배열로 감싸기
            if isinstance(result, dict):
                return [result]
            return result
        except json.JSONDecodeError:
            # Groq이 줄바꿈으로 이어진 복수 JSON 객체를 반환한 경우 처리
            items = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    continue
            if items:
                return items
            return []

    except Exception as e:
        err_body = e.response.text if hasattr(e, 'response') else str(e)
        print(f"[Groq] {model} 실패: {e} | 본문: {err_body[:200]}")
        raise


async def parse_edit_intent(text: str) -> dict:
    """수정 의도 파싱 — search(기존 제목 키워드), 변경할 필드 반환"""
    kst = timezone(timedelta(hours=9))
    today_str = datetime.now(kst).strftime("%Y-%m-%d")

    system_prompt = f"""오늘 날짜: {today_str} (KST)

사용자가 기존 일정을 수정하려 합니다. 아래 텍스트에서 수정 정보를 추출하세요.
상대적 날짜는 오늘 기준 절대 날짜로 변환하세요.

반드시 JSON 객체만 반환하세요. 마크다운 펜스 없이 순수 JSON만.
형식:
{{
  "search": "기존 일정 제목에서 찾을 키워드 (짧게)",
  "title":    "새 제목 (변경 없으면 null)",
  "datetime": "새 날짜시간 YYYY-MM-DDTHH:MM (변경 없으면 null)",
  "memo":     "새 메모 (변경 없으면 null)"
}}"""

    payload = {
        "model":    GROQ_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text}
        ],
        "temperature": 0.1,
        "max_tokens":  500,
    }

    async with httpx.AsyncClient() as c:
        resp = await c.post(
            GROQ_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

    raw = data["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def is_edit_intent(text: str) -> bool:
    """수정/변경 의도 키워드 감지"""
    keywords = ["바꿔", "변경", "수정", "고쳐", "옮겨", "미뤄", "앞당겨",
                "바꿔줘", "변경해줘", "수정해줘", "고쳐줘", "옮겨줘"]
    return any(kw in text for kw in keywords)


# ─────────────────────────────────────────
#  리마인더 스케줄러
# ─────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone="Asia/Seoul")

REMINDER_LABELS = {
    "3d": "⏰ 3일 전",
    "1d": "🔔 내일",
    "3h": "🚨 3시간 전",
}

async def check_reminders():
    """10분마다 실행 — 3일 전 / 1일 전 / 3시간 전 알림"""
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
                ("3d", 3*24*60, 2.5*24*60),
                ("1d", 1*24*60, 0.5*24*60),
                ("3h", 3*60,    2.5*60),
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

    except Exception as e:
        print(f"[리마인더 오류] {e}")


async def morning_briefing():
    """매일 오전 9시 — 오늘 일정 + 진행 중 기간 일정 + 3일 후 일정 통합 브리핑"""
    try:
        kst   = timezone(timedelta(hours=9))
        today = datetime.now(kst).date()
        three_days_later = today + timedelta(days=3)

        all_scheds = await db.schedules.find({"done": False}).to_list(300)

        today_list = [s for s in all_scheds
                      if s.get("datetime") and _safe_date(s["datetime"]) == today]

        soon_list = [s for s in all_scheds
                     if s.get("datetime") and _safe_date(s["datetime"]) == three_days_later]

        range_list = []
        for s in all_scheds:
            if not s.get("start_date") or not s.get("end_date"):
                continue
            try:
                start_d = datetime.fromisoformat(s["start_date"]).date()
                end_d   = datetime.fromisoformat(s["end_date"]).date()
            except Exception:
                continue
            if not (start_d <= today <= end_d):
                continue
            if today.weekday() in s.get("exclude_weekdays", []):
                continue
            range_list.append(s)

        if not today_list and not soon_list and not range_list:
            await send_telegram(
                f"🌅 <b>오늘 브리핑 ({today.strftime('%m월 %d일')})</b>\n\n✨ 오늘 예정된 일정이 없습니다.")
            return

        lines = [f"🌅 <b>오늘 브리핑 ({today.strftime('%m월 %d일')})</b>\n"]
        inline_buttons = []

        if today_list:
            lines.append("📅 <b>오늘 일정</b>")
            for s in today_list:
                dt_label = ""
                try:
                    dt_label = f"  {datetime.fromisoformat(s['datetime']).strftime('%H:%M')}"
                except Exception:
                    pass
                lines.append(f"• <b>{s['title']}</b>{dt_label}")
                inline_buttons.append([{
                    "text": f"✅ {s['title'][:20]}",
                    "callback_data": f"done_{str(s['_id'])}"
                }])

        if range_list:
            lines.append("\n🗓 <b>진행 중인 일정</b>")
            for s in range_list:
                try:
                    start_d   = datetime.fromisoformat(s["start_date"]).date()
                    end_d     = datetime.fromisoformat(s["end_date"]).date()
                    day_idx   = (today - start_d).days + 1
                    days_left = (end_d - today).days
                    lines.append(f"• <b>{s['title']}</b>  {day_idx}일째 / {days_left}일 남음")
                except Exception:
                    lines.append(f"• <b>{s['title']}</b>")
                inline_buttons.append([{
                    "text": f"✅ {s['title'][:20]} 완료",
                    "callback_data": f"done_{str(s['_id'])}"
                }])

        if soon_list:
            lines.append(f"\n⏰ <b>3일 후 ({three_days_later.strftime('%m/%d')})</b>")
            for s in soon_list:
                dt_label = ""
                try:
                    dt_label = f"  {datetime.fromisoformat(s['datetime']).strftime('%H:%M')}"
                except Exception:
                    pass
                lines.append(f"• <b>{s['title']}</b>{dt_label}")
                inline_buttons.append([{
                    "text": f"⏰ {s['title'][:20]}",
                    "callback_data": f"done_{str(s['_id'])}"
                }])

        markup = {"inline_keyboard": inline_buttons} if inline_buttons else None
        await send_telegram("\n".join(lines), markup)

    except Exception as e:
        print(f"[브리핑 오류] {e}")


# ─────────────────────────────────────────
#  Lifespan
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(check_reminders,  "interval", minutes=10,          id="reminders")
    scheduler.add_job(morning_briefing, "cron",     hour=9, minute=0,    id="morning_briefing")
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
    start_date: Optional[str] = None        # 기간 일정 시작일 YYYY-MM-DD
    end_date: Optional[str] = None          # 기간 일정 종료일 YYYY-MM-DD
    exclude_weekdays: Optional[list] = []   # 제외 요일 [0=월 ~ 6=일]
    repeat: Optional[str] = None            # "weekly" | "monthly"

class ScheduleUpdate(BaseModel):
    title:    Optional[str] = None
    datetime: Optional[str] = None
    memo:     Optional[str] = None

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
        "title":            sched.title,
        "datetime":         sched.datetime,
        "memo":             sched.memo or "",
        "start_date":       sched.start_date,
        "end_date":         sched.end_date,
        "exclude_weekdays": sched.exclude_weekdays or [],
        "repeat":           sched.repeat,
        "done":             False,
        "notified":         [],
        "created_at":       datetime.utcnow().isoformat()
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

@app.put("/api/schedules/{id}")
async def update_schedule(id: str, sched: ScheduleUpdate):
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID")
    fields = {k: v for k, v in sched.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="변경할 내용이 없습니다.")
    result = await db.schedules.update_one({"_id": oid}, {"$set": fields})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="일정을 찾을 수 없습니다.")
    updated = await db.schedules.find_one({"_id": oid})
    return sched_to_dict(updated)


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

        async def answer_cq(text=""):
            async with httpx.AsyncClient() as c:
                await c.post(f"{TELEGRAM_API}/answerCallbackQuery",
                             json={"callback_query_id": cq_id, "text": text})

        # ── 완료 처리
        if cq_data.startswith("done_"):
            sched_id = cq_data[5:]
            try:
                oid   = ObjectId(sched_id)
                sched = await db.schedules.find_one({"_id": oid})
                if sched:
                    await db.schedules.update_one({"_id": oid}, {"$set": {"done": True}})
                    repeat   = sched.get("repeat")
                    next_msg = ""
                    if repeat and sched.get("datetime"):
                        try:
                            dt = datetime.fromisoformat(sched["datetime"])
                            if repeat == "weekly":
                                next_dt = dt + timedelta(weeks=1)
                            elif repeat == "monthly":
                                month = dt.month + 1 if dt.month < 12 else 1
                                year  = dt.year  if dt.month < 12 else dt.year + 1
                                max_d = cal.monthrange(year, month)[1]
                                next_dt = dt.replace(year=year, month=month, day=min(dt.day, max_d))
                            else:
                                next_dt = None
                            if next_dt:
                                new_doc = {
                                    "title":      sched["title"],
                                    "datetime":   next_dt.isoformat()[:16],
                                    "memo":       sched.get("memo", ""),
                                    "repeat":     repeat,
                                    "done":       False,
                                    "notified":   [],
                                    "created_at": datetime.utcnow().isoformat()
                                }
                                await db.schedules.insert_one(new_doc)
                                next_msg = f"\n🔁 다음 회차 등록: {next_dt.strftime('%m월 %d일 %H:%M')}"
                        except Exception as e:
                            print(f"[반복 일정 오류] {e}")
                    await answer_cq("✅ 완료!")
                    await send_telegram(f"✅ <b>{sched['title']}</b> 완료 처리됐습니다.{next_msg}")
            except Exception as e:
                print(f"[콜백 오류] {e}")

        # ── 수정 확인
        elif cq_data.startswith("edit_confirm_"):
            pending_id = cq_data[13:]
            try:
                pending = await db.pending_edits.find_one({"_id": ObjectId(pending_id)})
                if not pending:
                    await answer_cq("⏰ 수정 요청이 만료됐습니다.")
                    return {"ok": True}
                oid    = ObjectId(pending["sched_id"])
                fields = {k: v for k, v in pending["fields"].items() if v is not None}
                await db.schedules.update_one({"_id": oid}, {"$set": fields})
                await db.pending_edits.delete_one({"_id": ObjectId(pending_id)})
                await answer_cq("✅ 수정됐습니다!")
                changes = []
                if fields.get("title"):    changes.append(f"제목: {fields['title']}")
                if fields.get("datetime"): changes.append(f"날짜: {datetime.fromisoformat(fields['datetime']).strftime('%m월 %d일 %H:%M')}")
                if fields.get("memo"):     changes.append(f"메모: {fields['memo']}")
                await send_telegram(f"✏️ <b>{pending['title']}</b> 수정 완료\n" + "\n".join(f"• {c}" for c in changes))
            except Exception as e:
                print(f"[수정 확인 오류] {e}")

        # ── 수정 취소
        elif cq_data.startswith("edit_cancel_"):
            pending_id = cq_data[12:]
            try:
                await db.pending_edits.delete_one({"_id": ObjectId(pending_id)})
                await answer_cq("❌ 취소됐습니다.")
                await send_telegram("❌ 수정이 취소됐습니다.")
            except Exception as e:
                print(f"[수정 취소 오류] {e}")

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

    # ── 수정 의도 감지
    if text and is_edit_intent(text):
        try:
            intent = await parse_edit_intent(text)
            search_kw = intent.get("search", "").strip()
            if not search_kw:
                await send_telegram("❓ 어떤 일정을 수정할지 찾지 못했습니다.\n예: '거래처 미팅 목요일 3시로 바꿔줘'")
                return {"ok": True}

            # 제목에서 키워드 검색 (대소문자 무시)
            all_scheds = await db.schedules.find({"done": False}).to_list(200)
            matches = [s for s in all_scheds if search_kw in s.get("title", "")]
            if not matches:
                await send_telegram(f"🔍 <b>'{search_kw}'</b> 와 일치하는 일정을 찾지 못했습니다.\n/list 로 일정 목록을 확인해보세요.")
                return {"ok": True}

            sched = matches[0]  # 가장 먼저 매칭된 일정
            fields = {k: v for k, v in intent.items() if k != "search" and v is not None}

            # 변경 내용 미리보기
            lines = [f"✏️ <b>{sched['title']}</b> 을 수정할까요?\n"]
            if fields.get("title"):    lines.append(f"• 제목: {fields['title']}")
            if fields.get("datetime"):
                try:
                    lines.append(f"• 날짜: {datetime.fromisoformat(fields['datetime']).strftime('%m월 %d일 %H:%M')}")
                except Exception:
                    lines.append(f"• 날짜: {fields['datetime']}")
            if fields.get("memo"):     lines.append(f"• 메모: {fields['memo']}")

            # pending_edits에 임시 저장 (10분 TTL)
            expires = datetime.utcnow() + timedelta(minutes=10)
            pending = {
                "sched_id":  str(sched["_id"]),
                "title":     sched["title"],
                "fields":    fields,
                "expires_at": expires.isoformat(),
                "created_at": datetime.utcnow().isoformat()
            }
            result  = await db.pending_edits.insert_one(pending)
            pid     = str(result.inserted_id)

            markup = {"inline_keyboard": [[
                {"text": "✅ 확인",  "callback_data": f"edit_confirm_{pid}"},
                {"text": "❌ 취소",  "callback_data": f"edit_cancel_{pid}"}
            ]]}
            await send_telegram("\n".join(lines), markup)

        except Exception as e:
            print(f"[수정 파싱 오류] {e}")
            await send_telegram(f"❌ 수정 요청 처리 중 오류가 발생했습니다: {str(e)[:100]}")
        return {"ok": True}

    # ── 신규 일정 등록

    try:
        if photo:
            file_id = photo[-1]["file_id"]
            async with httpx.AsyncClient() as c:
                f_resp   = await c.get(f"{TELEGRAM_API}/getFile?file_id={file_id}")
                f_path   = f_resp.json()["result"]["file_path"]
                img_resp = await c.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{f_path}")
                image_bytes = img_resp.content
            try:
                parsed = await parse_with_groq(image_bytes=image_bytes)
            except Exception:
                await send_telegram("⚠️ 이미지 파싱에 실패했습니다.\n일정 내용을 텍스트로 다시 보내주세요.")
                return {"ok": True}
        else:
            parsed = await parse_with_groq(text=text)

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
