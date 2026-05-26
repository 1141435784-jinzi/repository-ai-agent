"""模拟支付 API 工具实现"""

import asyncio
import uuid
import time
from typing import Any, ClassVar, Dict, Optional, Type
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from src.tools.base import BaseTool
from src.tools.registry import register_tool


class PaymentMethod(str, Enum):
    """支付方式枚举"""
    WECHAT = "wechat"
    ALIPAY = "alipay"


class PaymentStatus(str, Enum):
    """支付状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class MockPaymentDatabase:
    """模拟支付数据库"""

    def __init__(self):
        self.orders = {}
        self.transactions = {}
        self._lock = asyncio.Lock()

    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            order_id = str(uuid.uuid4())[:8]
            transaction_id = f"T{int(time.time() * 1000)}"

            order = {
                "order_id": order_id,
                "transaction_id": transaction_id,
                "amount": order_data["amount"],
                "currency": order_data.get("currency", "CNY"),
                "payment_method": order_data["payment_method"],
                "description": order_data.get("description", ""),
                "status": PaymentStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "expire_time": (datetime.now() + timedelta(minutes=30)).isoformat(),
                "notify_url": order_data.get("notify_url"),
                "return_url": order_data.get("return_url"),
                "metadata": order_data.get("metadata", {}),
                "user_id": order_data.get("user_id", "test_user"),
            }

            self.orders[order_id] = order
            self.transactions[transaction_id] = {
                "transaction_id": transaction_id,
                "order_id": order_id,
                "status": PaymentStatus.PENDING,
                "created_at": order["created_at"]
            }

            return order

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        return self.orders.get(order_id)

    async def update_order_status(self, order_id: str, status: PaymentStatus) -> bool:
        async with self._lock:
            if order_id in self.orders:
                self.orders[order_id]["status"] = status
                self.orders[order_id]["updated_at"] = datetime.now().isoformat()
                transaction_id = self.orders[order_id]["transaction_id"]
                if transaction_id in self.transactions:
                    self.transactions[transaction_id]["status"] = status
                    self.transactions[transaction_id]["updated_at"] = datetime.now().isoformat()
                return True
            return False

    async def process_payment(self, order_id: str) -> Dict[str, Any]:
        async with self._lock:
            if order_id not in self.orders:
                return {"error": True, "message": "订单不存在"}

            order = self.orders[order_id]
            await asyncio.sleep(1)

            import random
            success = random.random() < 1

            status = PaymentStatus.SUCCESS if success else PaymentStatus.FAILED
            message = "支付成功" if success else "支付失败（模拟）"

            order["status"] = status
            order["updated_at"] = datetime.now().isoformat()
            order["paid_at"] = datetime.now().isoformat() if success else None

            transaction_id = order["transaction_id"]
            self.transactions[transaction_id]["status"] = status
            self.transactions[transaction_id]["updated_at"] = datetime.now().isoformat()
            self.transactions[transaction_id]["completed_at"] = datetime.now().isoformat() if success else None

            return {
                "success": success,
                "order_id": order_id,
                "transaction_id": transaction_id,
                "status": status,
                "message": message,
                "paid_amount": order["amount"] if success else 0,
                "paid_at": order.get("paid_at")
            }

    async def refund_order(self, order_id: str, refund_amount: Optional[float] = None) -> Dict[str, Any]:
        async with self._lock:
            if order_id not in self.orders:
                return {"error": True, "message": "订单不存在"}

            order = self.orders[order_id]

            if order["status"] != PaymentStatus.SUCCESS:
                return {"error": True, "message": "只有支付成功的订单才能退款"}

            await asyncio.sleep(1)

            order["status"] = PaymentStatus.REFUNDED
            order["updated_at"] = datetime.now().isoformat()
            order["refunded_at"] = datetime.now().isoformat()
            order["refund_amount"] = refund_amount or order["amount"]

            return {
                "success": True,
                "order_id": order_id,
                "refund_amount": order["refund_amount"],
                "refunded_at": order["refunded_at"],
                "message": "退款成功（模拟）"
            }


_payment_db = MockPaymentDatabase()


class CreatePaymentInput(BaseModel):
    """创建支付订单输入参数"""
    amount: float = Field(gt=0, description="支付金额（单位：元）")
    payment_method: PaymentMethod = Field(description="支付方式：wechat（微信支付）或 alipay（支付宝）")
    description: Optional[str] = Field(default="商品购买", description="订单描述")
    currency: str = Field(default="CNY", description="货币代码")
    user_id: Optional[str] = Field(default="test_user", description="用户ID")


class CreatePaymentOutput(BaseModel):
    """创建支付订单输出"""
    success: bool = True
    message: Optional[str] = None
    order_id: str = Field(description="订单ID")
    transaction_id: str = Field(description="交易ID")
    amount: float = Field(description="支付金额")
    payment_method: str = Field(description="支付方式")
    qrcode_url: str = Field(description="支付二维码URL")


@register_tool
class CreatePaymentTool(BaseTool):
    name = "create_payment"
    description = "创建支付订单，支持微信支付和支付宝支付（模拟）"
    args_schema: ClassVar[Type[BaseModel]] = CreatePaymentInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "payment", "tags": ["payment", "wechat", "alipay"]}

    async def _arun(self, amount: float, payment_method: PaymentMethod,
                    description: str = "商品购买", currency: str = "CNY",
                    user_id: str = "test_user") -> CreatePaymentOutput:
        order_data = {
            "amount": amount,
            "payment_method": payment_method,
            "description": description,
            "currency": currency,
            "user_id": user_id,
            "metadata": {}
        }

        order = await _payment_db.create_order(order_data)

        if payment_method == PaymentMethod.WECHAT:
            qrcode_url = f"https://api.payment-mock.com/wechat/qrcode/{order['order_id']}"
        else:
            qrcode_url = f"https://api.payment-mock.com/alipay/qrcode/{order['order_id']}"

        return CreatePaymentOutput(
            success=True,
            message="支付订单创建成功",
            order_id=order["order_id"],
            transaction_id=order["transaction_id"],
            amount=amount,
            payment_method=payment_method,
            qrcode_url=qrcode_url
        )


class ProcessPaymentInput(BaseModel):
    """处理支付输入参数"""
    order_id: str = Field(description="订单ID")


class ProcessPaymentOutput(BaseModel):
    """处理支付输出"""
    success: bool = True
    message: Optional[str] = None
    order_id: str = Field(description="订单ID")
    transaction_id: str = Field(description="交易ID")
    status: str = Field(description="支付状态")
    paid_amount: float = Field(description="支付金额")


@register_tool
class ProcessPaymentTool(BaseTool):
    name = "process_payment"
    description = "处理支付（模拟支付流程）"
    args_schema: ClassVar[Type[BaseModel]] = ProcessPaymentInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "payment", "tags": ["payment", "process"]}

    async def _arun(self, order_id: str) -> ProcessPaymentOutput:
        order = await _payment_db.get_order(order_id)
        if not order:
            return ProcessPaymentOutput(
                success=False,
                message="订单不存在",
                order_id=order_id
            )

        if order["status"] != PaymentStatus.PENDING:
            return ProcessPaymentOutput(
                success=False,
                message=f"订单状态为 {order['status']}，无法处理支付",
                order_id=order_id
            )

        await _payment_db.update_order_status(order_id, PaymentStatus.PROCESSING)

        result = await _payment_db.process_payment(order_id)

        if result.get("success"):
            return ProcessPaymentOutput(
                success=True,
                message=result["message"],
                order_id=result["order_id"],
                transaction_id=result["transaction_id"],
                status=result["status"],
                paid_amount=result["paid_amount"]
            )
        else:
            return ProcessPaymentOutput(
                success=False,
                message=result.get("message", "支付失败"),
                order_id=order_id
            )


class QueryPaymentInput(BaseModel):
    """查询支付状态输入参数"""
    order_id: str = Field(description="订单ID")


class QueryPaymentOutput(BaseModel):
    """查询支付状态输出"""
    success: bool = True
    message: Optional[str] = None
    order_id: str = Field(description="订单ID")
    transaction_id: str = Field(description="交易ID")
    status: str = Field(description="支付状态")
    amount: float = Field(description="支付金额")
    paid_at: Optional[str] = Field(description="支付时间")


@register_tool
class QueryPaymentTool(BaseTool):
    name = "query_payment"
    description = "查询支付状态"
    args_schema: ClassVar[Type[BaseModel]] = QueryPaymentInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "payment", "tags": ["payment", "query"]}

    async def _arun(self, order_id: str) -> QueryPaymentOutput:
        order = await _payment_db.get_order(order_id)
        if not order:
            return QueryPaymentOutput(
                success=False,
                message="订单不存在",
                order_id=order_id
            )

        expire_time = datetime.fromisoformat(order["expire_time"])
        is_expired = datetime.now() > expire_time

        if is_expired and order["status"] == PaymentStatus.PENDING:
            await _payment_db.update_order_status(order_id, PaymentStatus.CANCELLED)
            order["status"] = PaymentStatus.CANCELLED

        return QueryPaymentOutput(
            success=True,
            message="查询成功",
            order_id=order["order_id"],
            transaction_id=order["transaction_id"],
            status=order["status"],
            amount=order["amount"],
            paid_at=order.get("paid_at")
        )


class RefundPaymentInput(BaseModel):
    """退款输入参数"""
    order_id: str = Field(description="订单ID")
    refund_amount: Optional[float] = Field(default=None, description="退款金额，不填则全额退款")


class RefundPaymentOutput(BaseModel):
    """退款输出"""
    success: bool = True
    message: Optional[str] = None
    order_id: str = Field(description="订单ID")
    refund_amount: float = Field(description="退款金额")
    refunded_at: str = Field(description="退款时间")


@register_tool
class RefundPaymentTool(BaseTool):
    name = "refund_payment"
    description = "退款（模拟退款流程）"
    args_schema: ClassVar[Type[BaseModel]] = RefundPaymentInput
    metadata: ClassVar[Optional[Dict[str, Any]]] = {"category": "payment", "tags": ["payment", "refund"]}

    async def _arun(self, order_id: str, refund_amount: Optional[float] = None) -> RefundPaymentOutput:
        result = await _payment_db.refund_order(order_id, refund_amount)

        if result.get("success"):
            return RefundPaymentOutput(
                success=True,
                message=result["message"],
                order_id=result["order_id"],
                refund_amount=result["refund_amount"],
                refunded_at=result["refunded_at"]
            )
        else:
            return RefundPaymentOutput(
                success=False,
                message=result.get("message", "退款失败"),
                order_id=order_id
            )