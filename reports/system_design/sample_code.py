from typing import List, Dict, Tuple


class IUserPreferences:
    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        pass


class IContentInfo:
    def get_content_info(self, content_ids: List[str]) -> Dict[str, Tuple[str, float]]:
        pass


class IStrategy:
    def calculate_priority(
        self,
        content_id: str,
        user_preferences: Dict[str, float],
        content_info: Dict[str, Tuple[str, float]]
    ) -> float:
        pass


class ContentReorderService:
    def reorder_content(
        self,
        user_id: str,
        content_ids: List[str],
        user_preferences_source: IUserPreferences,
        content_info_source: IContentInfo,
        strategy: IStrategy
    ) -> List[str]:
        pass

    def _validate_inputs(
        self,
        user_preferences: Dict[str, float],
        content_info: Dict[str, Tuple[str, float]]
    ) -> None:
        pass

    def _sort_contents(
        self, scored_contents: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        pass
