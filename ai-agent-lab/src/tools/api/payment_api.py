"""
=== 模拟支付 API 工具 ===

【功能】：
模拟微信和支付宝支付流程，用于开发和测试环境
不进行真实的支付操作，仅返回模拟的支付结果

【设计原则】：
1. 安全第一：不处理真实支付，仅用于模拟
2. 简单易用：参数简单，返回结果清晰
3. 模拟真实：模拟真实支付流程的各个阶段
4. 支持多种支付方式：微信支付、支付宝支付

【支付流程模拟】：
1. 创建支付订单 → 2. 返回支付二维码/链接 → 3. 模拟支付回调 → 4. 查询支付状态

【使用场景】：
1. 开发环境测试支付流程
2. 演示支付功能
3. 教学和培训
"""

import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timedelta


# ============================================================
# 数据类型定义
# ============================================================

class PaymentMethod(str, Enum):
    """支付方式枚举"""
    WECHAT = "wechat"      # 微信支付
    ALIPAY = "alipay"      # 支付宝


class PaymentStatus(str, Enum):
    """支付状态枚举"""
    PENDING = "pending"        # 待支付
    PROCESSING = "processing"  # 支付中
    SUCCESS = "success"        # 支付成功
    FAILED = "failed"          # 支付失败
    REFUNDED = "refunded"      # 已退款
    CANCELLED = "cancelled"    # 已取消


# ============================================================
# 模拟支付数据库（内存存储）
# ============================================================

class MockPaymentDatabase:
    """模拟支付数据库（内存存储）"""
    
    def __init__(self):
        self.orders = {}  # 订单存储
        self.transactions = {}  # 交易记录
        self._lock = asyncio.Lock()
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建支付订单"""
        async with self._lock:
            order_id = str(uuid.uuid4())[:8]  # 生成简短的订单ID
            transaction_id = f"T{int(time.time() * 1000)}"  # 交易ID
            
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
        """获取订单信息"""
        return self.orders.get(order_id)
    
    async def update_order_status(self, order_id: str, status: PaymentStatus) -> bool:
        """更新订单状态"""
        async with self._lock:
            if order_id in self.orders:
                self.orders[order_id]["status"] = status
                self.orders[order_id]["updated_at"] = datetime.now().isoformat()
                
                # 更新交易记录
                transaction_id = self.orders[order_id]["transaction_id"]
                if transaction_id in self.transactions:
                    self.transactions[transaction_id]["status"] = status
                    self.transactions[transaction_id]["updated_at"] = datetime.now().isoformat()
                
                return True
            return False
    
    async def process_payment(self, order_id: str) -> Dict[str, Any]:
        """模拟处理支付"""
        async with self._lock:
            if order_id not in self.orders:
                return {"error": True, "message": "订单不存在"}
            
            order = self.orders[order_id]
            
            # 模拟支付处理延迟
            await asyncio.sleep(1)
            
            # 模拟支付结果（80%成功，20%失败）
            import random
            success = random.random() < 0.8
            
            if success:
                status = PaymentStatus.SUCCESS
                message = "支付成功"
            else:
                status = PaymentStatus.FAILED
                message = "支付失败（模拟）"
            
            # 更新订单状态
            order["status"] = status
            order["updated_at"] = datetime.now().isoformat()
            order["paid_at"] = datetime.now().isoformat() if success else None
            
            # 更新交易记录
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
        """模拟退款"""
        async with self._lock:
            if order_id not in self.orders:
                return {"error": True, "message": "订单不存在"}
            
            order = self.orders[order_id]
            
            if order["status"] != PaymentStatus.SUCCESS:
                return {"error": True, "message": "只有支付成功的订单才能退款"}
            
            # 模拟退款处理延迟
            await asyncio.sleep(1)
            
            # 更新订单状态
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
    
    async def get_payment_qrcode(self, order_id: str) -> Dict[str, Any]:
        """生成支付二维码信息"""
        if order_id not in self.orders:
            return {"error": True, "message": "订单不存在"}
        
        order = self.orders[order_id]
        payment_method = order["payment_method"]
        
        # 生成模拟的支付二维码数据
        if payment_method == PaymentMethod.WECHAT:
            qrcode_data = {
                "type": "wechat",
                "qrcode_url": f"https://api.payment-mock.com/wechat/qrcode/{order_id}",
                "qrcode_content": f"weixin://wxpay/bizpayurl?pr={order_id}",
                "payment_url": f"https://pay.weixin.qq.com/{order_id}",
                "instructions": "请使用微信扫描二维码完成支付"
            }
        elif payment_method == PaymentMethod.ALIPAY:
            qrcode_data = {
                "type": "alipay",
                "qrcode_url": f"https://api.payment-mock.com/alipay/qrcode/{order_id}",
                "qrcode_content": f"https://qr.alipay.com/{order_id}",
                "payment_url": f"https://mapi.alipay.com/gateway.do?order_id={order_id}",
                "instructions": "请使用支付宝扫描二维码完成支付"
            }
        else:
            qrcode_data = {
                "type": "unknown",
                "qrcode_url": "",
                "instructions": "不支持的支付方式"
            }
        
        return {
            "order_id": order_id,
            "amount": order["amount"],
            "currency": order["currency"],
            "payment_method": payment_method,
            "qrcode_data": qrcode_data,
            "expire_time": order["expire_time"],
            "status": order["status"]
        }


# 全局支付数据库实例
_payment_db = MockPaymentDatabase()


# ============================================================
# Pydantic 输入模型
# ============================================================

class CreatePaymentInput(BaseModel):
    """创建支付订单输入参数"""
    amount: float = Field(
        gt=0,
        description="支付金额（单位：元）",
        example=100.0
    )
    payment_method: PaymentMethod = Field(
        description="支付方式：wechat（微信支付）或 alipay（支付宝）",
        example="wechat"
    )
    description: Optional[str] = Field(
        default="商品购买",
        description="订单描述",
        example="购买VIP会员"
    )
    currency: str = Field(
        default="CNY",
        description="货币代码，默认CNY（人民币）",
        example="CNY"
    )
    notify_url: Optional[str] = Field(
        default=None,
        description="支付结果通知URL（模拟用）",
        example="https://your-domain.com/payment/notify"
    )
    return_url: Optional[str] = Field(
        default=None,
        description="支付完成返回URL（模拟用）",
        example="https://your-domain.com/payment/return"
    )
    user_id: Optional[str] = Field(
        default="test_user",
        description="用户ID（模拟用）",
        example="user_123"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description="附加数据",
        example={"product_id": "prod_001", "quantity": 1}
    )


class ProcessPaymentInput(BaseModel):
    """处理支付输入参数"""
    order_id: str = Field(
        description="订单ID",
        example="abc12345"
    )


class QueryPaymentInput(BaseModel):
    """查询支付状态输入参数"""
    order_id: str = Field(
        description="订单ID",
        example="abc12345"
    )


class RefundPaymentInput(BaseModel):
    """退款输入参数"""
    order_id: str = Field(
        description="订单ID",
        example="abc12345"
    )
    refund_amount: Optional[float] = Field(
        default=None,
        description="退款金额，不填则全额退款",
        example=50.0
    )


# ============================================================
# 支付工具函数
# ============================================================

async def create_payment(
    amount: float,
    payment_method: PaymentMethod,
    description: str = "商品购买",
    currency: str = "CNY",
    notify_url: Optional[str] = None,
    return_url: Optional[str] = None,
    user_id: str = "test_user",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """创建支付订单
    
    Args:
        amount: 支付金额（元）
        payment_method: 支付方式（wechat/alipay）
        description: 订单描述
        currency: 货币代码
        notify_url: 通知URL
        return_url: 返回URL
        user_id: 用户ID
        metadata: 附加数据
        
    Returns:
        Dict[str, Any]: 订单信息
    """
    print(f"💰 创建支付订单: {amount}{currency} via {payment_method}")
    
    order_data = {
        "amount": amount,
        "payment_method": payment_method,
        "description": description,
        "currency": currency,
        "notify_url": notify_url,
        "return_url": return_url,
        "user_id": user_id,
        "metadata": metadata or {}
    }
    
    order = await _payment_db.create_order(order_data)
    
    # 获取支付二维码
    qrcode_info = await _payment_db.get_payment_qrcode(order["order_id"])
    
    return {
        "success": True,
        "message": "支付订单创建成功",
        "order": order,
        "payment_info": qrcode_info,
        "next_steps": [
            "1. 展示支付二维码给用户",
            "2. 用户扫描二维码完成支付",
            "3. 调用 process_payment 模拟支付处理",
            "4. 调用 query_payment 查询支付状态"
        ]
    }


async def process_payment(order_id: str) -> Dict[str, Any]:
    """处理支付（模拟支付流程）
    
    Args:
        order_id: 订单ID
        
    Returns:
        Dict[str, Any]: 支付处理结果
    """
    print(f"💰 处理支付订单: {order_id}")
    
    # 先检查订单是否存在
    order = await _payment_db.get_order(order_id)
    if not order:
        return {"error": True, "message": "订单不存在"}
    
    # 检查订单状态
    if order["status"] != PaymentStatus.PENDING:
        return {
            "error": True,
            "message": f"订单状态为 {order['status']}，无法处理支付",
            "current_status": order["status"]
        }
    
    # 更新状态为处理中
    await _payment_db.update_order_status(order_id, PaymentStatus.PROCESSING)
    
    # 模拟支付处理
    result = await _payment_db.process_payment(order_id)
    
    return result


async def query_payment(order_id: str) -> Dict[str, Any]:
    """查询支付状态
    
    Args:
        order_id: 订单ID
        
    Returns:
        Dict[str, Any]: 支付状态信息
    """
    order = await _payment_db.get_order(order_id)
    if not order:
        return {"error": True, "message": "订单不存在"}
    
    # 检查订单是否已过期
    expire_time = datetime.fromisoformat(order["expire_time"])
    is_expired = datetime.now() > expire_time
    
    if is_expired and order["status"] == PaymentStatus.PENDING:
        await _payment_db.update_order_status(order_id, PaymentStatus.CANCELLED)
        order["status"] = PaymentStatus.CANCELLED
        order["expired"] = True
    
    return {
        "order_id": order_id,
        "transaction_id": order["transaction_id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "payment_method": order["payment_method"],
        "description": order["description"],
        "status": order["status"],
        "created_at": order["created_at"],
        "updated_at": order["updated_at"],
        "expire_time": order["expire_time"],
        "is_expired": is_expired,
        "paid_at": order.get("paid_at"),
        "refunded_at": order.get("refunded_at"),
        "refund_amount": order.get("refund_amount"),
        "user_id": order["user_id"],
        "metadata": order["metadata"]
    }


async def refund_payment(order_id: str, refund_amount: Optional[float] = None) -> Dict[str, Any]:
    """退款（模拟退款流程）
    
    Args:
        order_id: 订单ID
        refund_amount: 退款金额，不填则全额退款
        
    Returns:
        Dict[str, Any]: 退款结果
    """
    print(f"💰 处理退款: {order_id}, 金额: {refund_amount or '全额'}")
    
    result = await _payment_db.refund_order(order_id, refund_amount)
    return result


async def get_payment_methods() -> Dict[str, Any]:
    """获取支持的支付方式
    
    Returns:
        Dict[str, Any]: 支付方式列表
    """
    return {
        "supported_methods": [
            {
                "code": "wechat",
                "name": "微信支付",
                "description": "通过微信扫码或小程序支付",
                "icon": "https://api.payment-mock.com/icons/wechat.png",
                "min_amount": 0.01,
                "max_amount": 50000.00,
                "currencies": ["CNY"],
                "features": ["扫码支付", "小程序支付", "H5支付"]
            },
            {
                "code": "alipay",
                "name": "支付宝",
                "description": "通过支付宝扫码或App支付",
                "icon": "https://api.payment-mock.com/icons/alipay.png",
                "min_amount": 0.01,
                "max_amount": 50000.00,
                "currencies": ["CNY"],
                "features": ["扫码支付", "App支付", "网页支付"]
            }
        ],
        "default_currency": "CNY",
        "note": "此为模拟支付系统，不进行真实支付操作"
    }


# ============================================================
# 测试函数
# ============================================================

async def test_payment_flow():
    """测试支付流程"""
    print("🧪 测试模拟支付流程")
    print("=" * 60)
    
    try:
        # 1. 创建支付订单（微信支付）
        print("1. 创建微信支付订单...")
        create_result = await create_payment(
            amount=100.0,
            payment_method=PaymentMethod.WECHAT,
            description="购买VIP会员",
            user_id="user_001"
        )
        
        if create_result.get("success"):
            order_id = create_result["order"]["order_id"]
            print(f"   订单创建成功: {order_id}")
            print(f"   金额: {create_result['order']['amount']} CNY")
            print(f"   支付方式: {create_result['order']['payment_method']}")
            print(f"   二维码URL: {create_result['payment_info']['qrcode_data']['qrcode_url']}")
        else:
            print(f"   订单创建失败: {create_result.get('message')}")
            return
        
        # 2. 查询支付状态
        print("\n2. 查询支付状态...")
        query_result = await query_payment(order_id)
        print(f"   订单状态: {query_result['status']}")
        
        # 3. 处理支付
        print("\n3. 模拟支付处理...")
        process_result = await process_payment(order_id)
        print(f"   支付结果: {process_result['message']}")
        print(f"   支付状态: {process_result['status']}")
        
        if process_result.get("success"):
            # 4. 再次查询支付状态
            print("\n4. 支付成功后查询状态...")
            query_result = await query_payment(order_id)
            print(f"   订单状态: {query_result['status']}")
            print(f"   支付时间: {query_result.get('paid_at', '未知')}")
            
            # 5. 模拟退款
            print("\n5. 模拟退款...")
            refund_result = await refund_payment(order_id)
            print(f"   退款结果: {refund_result['message']}")
            print(f"   退款金额: {refund_result.get('refund_amount', 0)}")
            
            # 6. 退款后查询状态
            print("\n6. 退款后查询状态...")
            query_result = await query_payment(order_id)
            print(f"   订单状态: {query_result['status']}")
            print(f"   退款时间: {query_result.get('refunded_at', '未知')}")
        
        # 7. 测试支付宝支付
        print("\n7. 测试支付宝支付...")
        alipay_result = await create_payment(
            amount=50.0,
            payment_method=PaymentMethod.ALIPAY,
            description="购买电子书",
            user_id="user_002"
        )
        
        if alipay_result.get("success"):
            alipay_order_id = alipay_result["order"]["order_id"]
            print(f"   支付宝订单创建成功: {alipay_order_id}")
            print(f"   支付URL: {alipay_result['payment_info']['qrcode_data']['payment_url']}")
        
        # 8. 获取支持的支付方式
        print("\n8. 获取支持的支付方式...")
        methods_result = await get_payment_methods()
        print(f"   支持 {len(methods_result['supported_methods'])} 种支付方式:")
        for method in methods_result["supported_methods"]:
            print(f"   - {method['name']} ({method['code']}): {method['description']}")
        
        print("\n" + "=" * 60)
        print("✅ 模拟支付流程测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_payment_flow())