from korean_lunar_calendar import KoreanLunarCalendar

class SajuCalculator:
    def __init__(self):
        self.calendar = KoreanLunarCalendar()

    def analyze_five_elements(self, year, month, day):
        self.calendar.setSolarDate(year, month, day)
        # 실제로는 여기서 연/월/일/시의 간지를 가져와 오행을 매칭합니다.
        # 예시를 위해 1987년 1월 1일의 가상 데이터를 반환합니다.
        analysis_result = {
            "main_element": "정화", # 일간
            "counts": {"목": 1, "화": 2, "토": 2, "금": 1, "수": 2}
        }
        return analysis_result
    
    def analyze_five_elements(self, year, month, day):
    # 실제 계산 로직이 들어가는 곳
    # 지금은 테스트를 위해 가상의 데이터를 반환하도록 설정하세요.
        return {
        "main_element": "갑목", 
        "counts": {"목": 2, "화": 1, "토": 1, "금": 0, "수": 4}
    }