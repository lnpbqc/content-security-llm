"""从原业务后端读取管理员可选择的用户，不获取登录凭据。"""

from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field


class BusinessUsersUnavailableError(Exception):
    pass


class BusinessUsersTimeoutError(Exception):
    pass


class BusinessUser(BaseModel):
    """校验原用户接口的必要字段，其他私有字段不转发。"""

    model_config = ConfigDict(strict=True)
    id: int = Field(gt=0)
    username: str
    display_name: str
    role: str
    status: str


class BusinessUserPage(BaseModel):
    model_config = ConfigDict(strict=True)
    items: List[BusinessUser]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_pages: int = Field(ge=0)


class BusinessUsersResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    code: Literal[0]
    data: BusinessUserPage


class BusinessUserService:
    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/") + "/users"

    def list_users(self, page: int, page_size: int, keyword: Optional[str]):
        params = {"page": page, "page_size": page_size}
        if keyword is not None:
            params["keyword"] = keyword
        try:
            # 独立客户端，不复制调用者的 Bearer、管理员密钥或其他请求头。
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self.url, params=params)
                response.raise_for_status()
            result = BusinessUsersResponse.model_validate(response.json()).data
        except httpx.TimeoutException as exc:
            raise BusinessUsersTimeoutError("业务用户查询超时，请重试") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BusinessUsersUnavailableError("业务用户查询失败，请检查业务后端及用户接口") from exc
        return {
            "items": [{"id": user.id, "username": user.username,
                       "display_name": user.display_name, "role": user.role,
                       "enabled": user.status == "normal"} for user in result.items],
            "total": result.total, "page": result.page,
            "page_size": result.page_size, "total_pages": result.total_pages,
        }
