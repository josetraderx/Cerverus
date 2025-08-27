from fastapi import APIRouter

router = APIRouter(prefix="/ingest")


@router.post("/")
async def ingest(payload: dict):
    return {"ingested": True, "received": payload}
