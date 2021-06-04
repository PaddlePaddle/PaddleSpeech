import re
from .num import num2str


# 温度表达式，温度会影响负号的读法
# -3°C 零下三度
RE_TEMPERATURE = re.compile(
    r'(-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)'
)
def replace_temperature(match: re.Match) -> str:
    sign = match.group(1)
    temperature = match.group(2)
    unit = match.group(3)
    sign: str = "零下" if sign else ""
    temperature: str = num2str(temperature)
    unit: str = "摄氏度" if unit == "摄氏度" else "度"
    result = f"{sign}{temperature}{unit}"
    return result
