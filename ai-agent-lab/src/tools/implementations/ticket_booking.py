"""票务预订工具实现"""

import asyncio
import uuid
import time
from typing import Any, ClassVar, Dict, Optional, Type
from enum import Enum
from pydantic import BaseModel, Field

from src.tools.base import BaseTool
from src.tools.registry import register_tool


class BookingStatus(str, Enum):
    """预订状态枚举"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class MockTicketDB:
    """模拟票务数据库"""

    def __init__(self):
        self.bookings = {}
        self._lock = asyncio.Lock()

    async def create_booking(self, ticket_type: str, data: Dict) -> Dict:
        booking_id = f"{ticket_type[0].upper()}BK{int(time.time() * 1000)}"
        ticket_number = f"TK{uuid.uuid4().hex[:8].upper()}"

        booking = {
            "booking_id": booking_id,
            "ticket_number": ticket_number,
            "ticket_type": ticket_type,
            "status": BookingStatus.PENDING,
            "departure_city": data["departure_city"],
            "arrival_city": data["arrival_city"],
            "departure_date": data["departure_date"],
            "departure_time": data["departure_time"],
            "arrival_time": data["arrival_time"],
            "flight_number": data.get("flight_number"),
            "train_number": data.get("train_number"),
            "passenger_name": data.get("passenger_name", "张三"),
            "passenger_id": data.get("passenger_id", "110101199001011234"),
            "price": data["price"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.bookings[booking_id] = booking
        return booking

    async def get_booking(self, booking_id: str) -> Optional[Dict]:
        return self.bookings.get(booking_id)

    async def cancel_booking(self, booking_id: str) -> bool:
        if booking_id in self.bookings:
            self.bookings[booking_id]["status"] = BookingStatus.CANCELLED
            self.bookings[booking_id]["cancelled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False


_db = MockTicketDB()


class FlightBookingInput(BaseModel):
    """机票预订输入"""
    departure_city: str = Field(description="出发城市")
    arrival_city: str = Field(description="到达城市")
    departure_date: str = Field(description="出发日期，格式：YYYY-MM-DD")
    departure_time: str = Field(description="出发时间，格式：HH:MM")
    arrival_time: str = Field(description="到达时间，格式：HH:MM")
    flight_number: str = Field(description="航班号")
    price: float = Field(description="票价")
    passenger_name: str = Field(default="张三", description="乘客姓名")
    passenger_id: str = Field(default="110101199001011234", description="证件号码")


class FlightBookingOutput(BaseModel):
    """机票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    ticket_number: str = Field(description="票号")
    flight_number: str = Field(description="航班号")
    passenger_name: str = Field(description="乘客姓名")
    price: float = Field(description="票价")


@register_tool
class FlightBookingTool(BaseTool):
    name = "book_flight"
    description = "预订机票，支持国内主要城市"
    args_schema: ClassVar[Type[BaseModel]] = FlightBookingInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["flight", "booking", "travel"]}

    async def _arun(self, departure_city: str, arrival_city: str, departure_date: str,
                    departure_time: str, arrival_time: str, flight_number: str,
                    price: float, passenger_name: str = "张三",
                    passenger_id: str = "110101199001011234") -> FlightBookingOutput:
        booking = await _db.create_booking("flight", {
            "departure_city": departure_city,
            "arrival_city": arrival_city,
            "departure_date": departure_date,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "flight_number": flight_number,
            "price": price,
            "passenger_name": passenger_name,
            "passenger_id": passenger_id
        })

        return FlightBookingOutput(
            success=True,
            message="机票预订成功",
            booking_id=booking["booking_id"],
            ticket_number=booking["ticket_number"],
            flight_number=flight_number,
            passenger_name=passenger_name,
            price=price
        )


class CancelFlightInput(BaseModel):
    """取消机票预订输入"""
    booking_id: str = Field(description="预订订单ID")


class CancelFlightOutput(BaseModel):
    """取消机票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    refund_amount: float = Field(description="退款金额")


@register_tool
class CancelFlightTool(BaseTool):
    name = "cancel_flight"
    description = "取消机票预订"
    args_schema: ClassVar[Type[BaseModel]] = CancelFlightInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["flight", "cancel", "refund"]}

    async def _arun(self, booking_id: str) -> CancelFlightOutput:
        booking = await _db.get_booking(booking_id)
        if not booking:
            return CancelFlightOutput(
                success=False,
                message="订单不存在",
                booking_id=booking_id
            )

        if booking["ticket_type"] != "flight":
            return CancelFlightOutput(
                success=False,
                message="不是机票订单",
                booking_id=booking_id
            )

        await _db.cancel_booking(booking_id)

        return CancelFlightOutput(
            success=True,
            message="机票退票成功",
            booking_id=booking_id,
            refund_amount=booking["price"]
        )


class QueryFlightInput(BaseModel):
    """查询机票预订输入"""
    booking_id: str = Field(description="预订订单ID")


class QueryFlightOutput(BaseModel):
    """查询机票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    ticket_number: str = Field(description="票号")
    flight_number: str = Field(description="航班号")
    status: str = Field(description="订单状态")
    departure_city: str = Field(description="出发城市")
    arrival_city: str = Field(description="到达城市")
    departure_date: str = Field(description="出发日期")
    departure_time: str = Field(description="出发时间")


@register_tool
class QueryFlightTool(BaseTool):
    name = "query_flight"
    description = "查询机票预订状态"
    args_schema: ClassVar[Type[BaseModel]] = QueryFlightInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["flight", "query"]}

    async def _arun(self, booking_id: str) -> QueryFlightOutput:
        booking = await _db.get_booking(booking_id)
        if not booking:
            return QueryFlightOutput(
                success=False,
                message="订单不存在",
                booking_id=booking_id
            )

        if booking["ticket_type"] != "flight":
            return QueryFlightOutput(
                success=False,
                message="不是机票订单",
                booking_id=booking_id
            )

        return QueryFlightOutput(
            success=True,
            message="查询成功",
            booking_id=booking["booking_id"],
            ticket_number=booking["ticket_number"],
            flight_number=booking["flight_number"],
            status=booking["status"],
            departure_city=booking["departure_city"],
            arrival_city=booking["arrival_city"],
            departure_date=booking["departure_date"],
            departure_time=booking["departure_time"]
        )


class TrainBookingInput(BaseModel):
    """高铁票预订输入"""
    departure_city: str = Field(description="出发城市")
    arrival_city: str = Field(description="到达城市")
    departure_date: str = Field(description="出发日期，格式：YYYY-MM-DD")
    departure_time: str = Field(description="出发时间，格式：HH:MM")
    arrival_time: str = Field(description="到达时间，格式：HH:MM")
    train_number: str = Field(description="车次")
    price: float = Field(description="票价")
    passenger_name: str = Field(default="张三", description="乘客姓名")
    passenger_id: str = Field(default="110101199001011234", description="证件号码")


class TrainBookingOutput(BaseModel):
    """高铁票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    ticket_number: str = Field(description="票号")
    train_number: str = Field(description="车次")
    passenger_name: str = Field(description="乘客姓名")
    price: float = Field(description="票价")


@register_tool
class TrainBookingTool(BaseTool):
    name = "book_train"
    description = "预订高铁票，支持国内主要城市"
    args_schema: ClassVar[Type[BaseModel]] = TrainBookingInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["train", "booking", "travel"]}

    async def _arun(self, departure_city: str, arrival_city: str, departure_date: str,
                    departure_time: str, arrival_time: str, train_number: str,
                    price: float, passenger_name: str = "张三",
                    passenger_id: str = "110101199001011234") -> TrainBookingOutput:
        booking = await _db.create_booking("train", {
            "departure_city": departure_city,
            "arrival_city": arrival_city,
            "departure_date": departure_date,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "train_number": train_number,
            "price": price,
            "passenger_name": passenger_name,
            "passenger_id": passenger_id
        })

        return TrainBookingOutput(
            success=True,
            message="高铁票预订成功",
            booking_id=booking["booking_id"],
            ticket_number=booking["ticket_number"],
            train_number=train_number,
            passenger_name=passenger_name,
            price=price
        )


class CancelTrainInput(BaseModel):
    """取消高铁票预订输入"""
    booking_id: str = Field(description="预订订单ID")


class CancelTrainOutput(BaseModel):
    """取消高铁票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    refund_amount: float = Field(description="退款金额")


@register_tool
class CancelTrainTool(BaseTool):
    name = "cancel_train"
    description = "取消高铁票预订"
    args_schema: ClassVar[Type[BaseModel]] = CancelTrainInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["train", "cancel", "refund"]}

    async def _arun(self, booking_id: str) -> CancelTrainOutput:
        booking = await _db.get_booking(booking_id)
        if not booking:
            return CancelTrainOutput(
                success=False,
                message="订单不存在",
                booking_id=booking_id
            )

        if booking["ticket_type"] != "train":
            return CancelTrainOutput(
                success=False,
                message="不是高铁票订单",
                booking_id=booking_id
            )

        await _db.cancel_booking(booking_id)

        return CancelTrainOutput(
            success=True,
            message="高铁票退票成功",
            booking_id=booking_id,
            refund_amount=booking["price"]
        )


class QueryTrainInput(BaseModel):
    """查询高铁票预订输入"""
    booking_id: str = Field(description="预订订单ID")


class QueryTrainOutput(BaseModel):
    """查询高铁票预订输出"""
    success: bool = True
    message: Optional[str] = None
    booking_id: str = Field(description="预订订单ID")
    ticket_number: str = Field(description="票号")
    train_number: str = Field(description="车次")
    status: str = Field(description="订单状态")
    departure_city: str = Field(description="出发城市")
    arrival_city: str = Field(description="到达城市")
    departure_date: str = Field(description="出发日期")
    departure_time: str = Field(description="出发时间")


@register_tool
class QueryTrainTool(BaseTool):
    name = "query_train"
    description = "查询高铁票预订状态"
    args_schema: ClassVar[Type[BaseModel]] = QueryTrainInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "booking", "tags": ["train", "query"]}

    async def _arun(self, booking_id: str) -> QueryTrainOutput:
        booking = await _db.get_booking(booking_id)
        if not booking:
            return QueryTrainOutput(
                success=False,
                message="订单不存在",
                booking_id=booking_id
            )

        if booking["ticket_type"] != "train":
            return QueryTrainOutput(
                success=False,
                message="不是高铁票订单",
                booking_id=booking_id
            )

        return QueryTrainOutput(
            success=True,
            message="查询成功",
            booking_id=booking["booking_id"],
            ticket_number=booking["ticket_number"],
            train_number=booking["train_number"],
            status=booking["status"],
            departure_city=booking["departure_city"],
            arrival_city=booking["arrival_city"],
            departure_date=booking["departure_date"],
            departure_time=booking["departure_time"]
        )