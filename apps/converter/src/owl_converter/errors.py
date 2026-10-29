"""结构化错误：所有拒启都必须带 code + 可执行的 hint。"""

from __future__ import annotations


class ConverterError(Exception):
    """转换器的唯一错误类型。

    exit_code 约定：
      2 = 输入/格式/方案不合法（默认）
      3 = 检查点不完整（仅 `--require-complete` 时）
      4 = 磁盘空间不足（仅 `--require-disk` 时）
      64 = 子命令尚未实现
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.exit_code = exit_code

    def __str__(self) -> str:
        text = f"[{self.code}] {self.message}"
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return text
