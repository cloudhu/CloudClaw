"""
SemVer 2.0.0 版本管理模块

遵循 https://semver.org/lang/zh-CN/
格式: MAJOR.MINOR.PATCH
- MAJOR: 不兼容的 API 修改
- MINOR: 向下兼容的功能新增
- PATCH: 向下兼容的问题修正
"""

import re
from dataclasses import dataclass
from typing import Optional


VERSION_STRING = "2.2.0"

SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@dataclass
class SemVer:
    """语义化版本号"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        if self.build:
            base += f"+{self.build}"
        return base

    def bump_major(self) -> "SemVer":
        """主版本号 +1，次版本号和修订号归零"""
        return SemVer(self.major + 1, 0, 0, self.prerelease, self.build)

    def bump_minor(self) -> "SemVer":
        """次版本号 +1，修订号归零"""
        return SemVer(self.major, self.minor + 1, 0, self.prerelease, self.build)

    def bump_patch(self) -> "SemVer":
        """修订号 +1"""
        return SemVer(self.major, self.minor, self.patch + 1, self.prerelease, self.build)

    def set_prerelease(self, tag: str) -> "SemVer":
        """设置预发布标签"""
        return SemVer(self.major, self.minor, self.patch, tag, self.build)

    def set_build(self, meta: str) -> "SemVer":
        """设置构建元数据"""
        return SemVer(self.major, self.minor, self.patch, self.prerelease, meta)

    def release(self) -> "SemVer":
        """移除预发布标签，转为正式版"""
        return SemVer(self.major, self.minor, self.patch)

    @classmethod
    def parse(cls, version: str) -> Optional["SemVer"]:
        """解析版本字符串"""
        m = SEMVER_PATTERN.match(version.strip())
        if not m:
            return None
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            prerelease=m.group("prerelease"),
            build=m.group("build"),
        )

    @classmethod
    def is_valid(cls, version: str) -> bool:
        """检查版本字符串是否合法"""
        return cls.parse(version) is not None

    def compare(self, other: "SemVer") -> int:
        """比较版本号，返回 1 / 0 / -1"""
        for a, b in [(self.major, other.major), (self.minor, other.minor), (self.patch, other.patch)]:
            if a > b:
                return 1
            if a < b:
                return -1
        # 有预发布标签的 < 无预发布标签的
        if self.prerelease and not other.prerelease:
            return -1
        if not self.prerelease and other.prerelease:
            return 1
        if self.prerelease and other.prerelease:
            # 简单字符串比较
            if self.prerelease > other.prerelease:
                return 1
            if self.prerelease < other.prerelease:
                return -1
        return 0

    def __eq__(self, other):
        return self.compare(other) == 0

    def __lt__(self, other):
        return self.compare(other) == -1

    def __gt__(self, other):
        return self.compare(other) == 1

    def __le__(self, other):
        return self.compare(other) <= 0

    def __ge__(self, other):
        return self.compare(other) >= 0


def get_version() -> str:
    """返回当前版本号字符串"""
    return VERSION_STRING


def get_semver() -> SemVer:
    """返回当前版本的 SemVer 对象"""
    return SemVer.parse(VERSION_STRING)
