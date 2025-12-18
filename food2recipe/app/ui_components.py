# File: food2recipe/app/ui_components.py
import streamlit as st


def _score_badge(score: float) -> str:
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def render_prediction_header(pred_food_key: str, score: float, title: str = None, raw_name: str = None):
    """
    A friendly header like a real food app.
    """
    dish_name = title or raw_name or pred_food_key or "món này"

    st.markdown(
        f"""
        <div class="card hero">
            <div class="kicker">Kết quả nhận diện</div>
            <div class="dish">🍽️ Đây là món <span class="dish-name">{dish_name}</span></div>
            <div class="meta">
                <span class="pill">🔎 Label: {pred_food_key}</span>
                <span class="pill green">✅ Độ tự tin: {_score_badge(score)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recipe_food_style(recipe: dict):
    """
    Render recipe in a friendly, food-style layout:
    - Two nice panels: Ingredients and Instructions
    - Natural Vietnamese wording
    """
    if not recipe:
        st.error("Mình chưa tìm thấy công thức cho món này trong file CSV.")
        return

    title = recipe.get("title")
    raw_name = recipe.get("food_name_raw")
    food_key = recipe.get("food_key") or recipe.get("food_name")

    dish_name = title or raw_name or food_key or "món này"

    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">📖 Công thức nấu {dish_name}</div>
            <div class="section-sub">Mình tóm tắt rõ ràng để bạn nấu theo từng bước dễ nhất.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="card panel">
                <div class="panel-title">🥕 Nguyên liệu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(recipe.get("ingredients", "Chưa có dữ liệu nguyên liệu."))

        st.markdown(
            """
            <div class="tip">
                Mẹo nhỏ: nếu bạn muốn món đậm vị hơn, hãy nêm từ từ rồi nếm lại trước khi tắt bếp.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card panel">
                <div class="panel-title">🍳 Cách nấu</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(recipe.get("instructions", "Chưa có dữ liệu cách nấu."))

        st.markdown(
            """
            <div class="tip">
                Nếu bạn nấu lần đầu, cứ làm đúng thứ tự các bước, món sẽ lên form rất ổn.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_topk(topk: list):
    """
    Show Top-k suggestions in a clean table-like list.
    topk item example: {"food_name": "banh_beo", "score": 0.89}
    """
    if not topk:
        return

    with st.expander("🔎 Xem các món tương tự (Top-k)", expanded=False):
        for item in topk:
            name = item.get("food_name", "unknown")
            score = item.get("score", None)

            st.markdown(
                f"""
                <div class="row">
                    <div class="row-left">🍲 {name}</div>
                    <div class="row-right">{_score_badge(score)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
