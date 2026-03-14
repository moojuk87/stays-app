from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
from icalendar import Calendar
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
import json
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSON 파일로 데이터 저장 (MongoDB 대신)
DB_FILE = "properties.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class Property(BaseModel):
    property_id: str
    name: str
    ical_url: str
    bookings: list = []
    last_synced: Optional[str] = None

# 숙소 전체 조회
@app.get("/api/properties")
async def get_properties():
    return load_db()

# 숙소 등록
@app.post("/api/properties")
async def create_property(prop: Property):
    props = load_db()
    # 중복 체크
    if any(p["property_id"] == prop.property_id for p in props):
        return prop
    props.append(prop.dict())
    save_db(props)
    return prop

# 숙소 삭제
@app.delete("/api/properties/{id}")
async def delete_property(id: str):
    props = load_db()
    new_props = [p for p in props if p["property_id"] != id]
    if len(new_props) == len(props):
        raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
    save_db(new_props)
    return {"status": "deleted"}

# iCal 동기화
@app.post("/api/properties/{id}/sync")
async def sync_ical(id: str):
    props = load_db()
    prop = next((p for p in props if p["property_id"] == id), None)
    if not prop:
        raise HTTPException(status_code=404, detail="숙소를 찾을 수 없습니다.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(prop["ical_url"], timeout=aiohttp.ClientTimeout(total=10)) as resp:
                ics_data = await resp.text()
        cal = Calendar.from_ical(ics_data)
        bookings = []
        for c in cal.walk("VEVENT"):
            start = c.get("dtstart")
            end = c.get("dtend")
            if start and end:
                bookings.append({
                    "start": str(start.dt),
                    "end": str(end.dt)
                })
        now = datetime.utcnow().isoformat()
        for p in props:
            if p["property_id"] == id:
                p["bookings"] = bookings
                p["last_synced"] = now
                break
        save_db(props)
        return {"status": "success", "bookings": bookings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동기화 실패: {str(e)}")

# 가용 숙소 검색
@app.get("/api/properties/search")
async def search_properties(start: datetime, end: datetime):
    props = load_db()
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
