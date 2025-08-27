from fastapi import APIRouter

router = APIRouter(prefix="/alerts")


@router.get("/")
async def list_alerts():
    return {"alerts": []}
