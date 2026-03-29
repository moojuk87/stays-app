from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
from icalendar import Calendar
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas 연결
MONGO_URL = os.environ.get("MONGO_URL")
client = AsyncIOMotorClient(MONGO_URL)
db = client.stays_db

class Property(BaseModel):
    property_id: str
    name: str
    ical_url: str
    bookings: list = []
    last_synced: Optional[str] = None

# 숙소 전체 조회
@app.get("/api/properties")
async def get_properties():
    props = await db.properties.find({}, {"_id": 0}).to_list(100)
    return props

# 숙소 등록
@app.post("/api/properties")
async def create_property(prop: Property):
    existing = await db.properties.find_one({"property_id": prop.property_id})
    if existing:
        return prop
    await db.properties.insert_one(prop.dict())
    return prop

# 숙소 삭제
@app.delete("/api/properties/{id}")
async def delete_property(id: str):
    result = await db.properties.delete_one({"property_id": id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
    return {"status": "deleted"}

# iCal 동기화
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
            # 1) 취소된 예약 제외
            status = str(c.get("status", "")).upper()
            if status == "CANCELLED":
                continue
            # 2) 호스트가 직접 막은 날짜 제외 ("Not available", "Blocked" 등)
            summary = str(c.get("summary", "")).strip()
            if any(kw in summary.lower() for kw in ["not available", "unavailable", "blocked", "준비중", "청소"]):
                continue
            start = c.get("dtstart")
            end = c.get("dtend")
            if start and end:
                # 3) 이미 끝난 과거 예약 제외
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

# 가용 숙소 검색
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
