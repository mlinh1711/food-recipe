# File: food2recipe/app/streamlit_app.py
import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports to work when running streamlit
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from food2recipe.core.settings import load_settings
from food2recipe.retrieval.recommender import RecipeRecommender
from food2recipe.retrieval.related_engine import SessionManager

from food2recipe.app.ui_components import render_recipe_food_style

# Page Config
st.set_page_config(
    page_title="Bếp Việt nè",
    page_icon="🍲",
    layout="wide",
)

# -------------------- CSS --------------------
st.markdown(
    """
    <style>
      .app-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
      }
      .app-title span {
        color: #ff6b6b;
      }
      .app-tagline {
        font-size: 16px;
        opacity: 0.85;
        margin-bottom: 18px;
      }
      .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 6px 22px rgba(0,0,0,0.25);
      }
      .meta {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 5px;
      }
      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        color: #eee;
      }
      .pill.green {
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.35);
        color: #2ecc71;
      }
      .rec-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #ff9f43;
      }
      .feedback-panel {
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.05);
      }
      .dish-name-hero {
          font-size: 20px;
          font-weight: 700;
          color: #ff6b6b;
      }
      .dish-name-sub {
          font-size: 18px;
          font-weight: 600;
          color: #feca57;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Resource Caching --------------------
@st.cache_resource
def get_recommender():
    settings = load_settings()
    rec = RecipeRecommender(settings)
    try:
        rec.load_resources()
        return rec
    except Exception as e:
        st.error(f"Không load được hệ thống: {e}")
        return None

def get_vietnamese_label(recommender, food_key):
    """
    Returns the Vietnamese name if available, otherwise a prettified text.
    Relies on settings.TITLE_COL = "vietnamese_name".
    """
    r = recommender.recipe_processor.get_recipe(food_key)
    if r and "title" in r:
        return r["title"]
    
    # Fallback: prettify the key (banh_mi -> Banh Mi)
    return food_key.replace("_", " ").title()

def render_stable_prediction_card(vn_name, food_key, score=None):
    score_html = ""
    if score is not None:
        score_html = f'<span class="pill green">✅ Độ tương đồng: {score:.2f}</span>'
        
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size: 13px; opacity: 0.7; margin-bottom: 4px;">Kết quả nhận diện (từ ảnh bạn tải lên)</div>
            <div class="dish-name-hero">{vn_name}</div>
            <div class="meta">
                <span class="pill">🏷️ Label: {food_key}</span>
                {score_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_current_view_card(vn_name, food_key):
    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid #feca57;">
            <div style="font-size: 13px; opacity: 0.7; margin-bottom: 4px;">Bạn đang xem</div>
            <div class="dish-name-sub">{vn_name}</div>
            <div class="meta">
                <span class="pill">🏷️ Label: {food_key}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    # -------- SESSION STATE --------
    if "liked_dishes" not in st.session_state:
        st.session_state.liked_dishes = set()
    if "disliked_dishes" not in st.session_state:
        st.session_state.disliked_dishes = set()
    if "last_upload_id" not in st.session_state:
        st.session_state.last_upload_id = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "current_view_item" not in st.session_state:
        st.session_state.current_view_item = None
    if "force_correct_item" not in st.session_state:
        st.session_state.force_correct_item = None
    if "show_correction_ui" not in st.session_state:
        st.session_state.show_correction_ui = False

    # -------- SIDEBAR --------
    st.sidebar.header("⚙️ Cài đặt")
    
    # 1. Personalization Toggle
    st.sidebar.markdown("### Cá nhân hóa")
    persist = st.sidebar.checkbox(
        "Cá nhân hoá trong phiên", 
        value=True,
        help="Giữ lại sở thích của bạn trong lúc dùng app."
    )
    st.sidebar.caption("Mình sẽ ưu tiên gợi ý theo món bạn thích trong lần dùng này nha.")
    
    if not persist:
        st.session_state.liked_dishes = set()
        st.session_state.disliked_dishes = set()
        
    # 2. Stats Block
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sở thích của bạn")
    
    liked_count = len(st.session_state.liked_dishes)
    blocked_count = len(st.session_state.disliked_dishes)
    
    st.sidebar.markdown(f"👍 **Đã thích:** {liked_count}")
    st.sidebar.markdown(f"🚫 **Không thích:** {blocked_count}")
    st.sidebar.caption("Bấm 👍/👎 ở phần kết quả để cập nhật liền.")

    # 3. Reset Button
    if st.sidebar.button("Đặt lại sở thích"):
        st.session_state.liked_dishes = set()
        st.session_state.disliked_dishes = set()
        st.toast("Đã đặt lại sở thích!")
        st.rerun()
    st.sidebar.caption("Xoá sở thích đã lưu trong phiên hiện tại.")

    # 4. Confidence Threshold
    st.sidebar.markdown("---")
    # Use 0-100 for display, convert to 0.0-1.0 for logic
    conf_percent = st.sidebar.slider(
        "Độ chắc chắn tối thiểu", 
        min_value=0, 
        max_value=100, 
        value=60, 
        step=5,
        format="%d%%"
    )
    confidence_threshold = conf_percent / 100.0
    
    st.sidebar.caption("Kéo cao → ít kết quả nhưng chắc hơn. Kéo thấp → nhiều gợi ý hơn.")

    # -------- APP TITLE --------
    st.markdown(
        """
        <div>
            <div class="app-title">🍲 <span>Bếp Việt nè</span></div>
            <div class="app-tagline">
                Bạn đưa mình một tấm ảnh món ăn nha. Mình đoán tên món,
                rồi chỉ bạn nguyên liệu và cách nấu liền tay luôn.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommender = get_recommender()
    if not recommender:
        st.warning("Hệ thống chưa sẵn sàng. Bạn nhớ chạy `python -m food2recipe.scripts.build_index` trước nha.")
        return
        
    recommender.settings.CONFIDENCE_THRESHOLD = confidence_threshold

    # -------- UPLOAD --------
    uploaded_file = st.file_uploader(
        "Gửi mình một tấm ảnh món ăn nha 🕵️‍♂️🍜",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        return

    # -------- PREDICTION LOGIC --------
    # Check if this is a new file
    file_id = uploaded_file.file_id if hasattr(uploaded_file, "file_id") else uploaded_file.name
    
    if file_id != st.session_state.last_upload_id:
        with st.spinner("Đang ngó nghiêng món ăn..."):
            try:
                result = recommender.predict(uploaded_file)
                st.session_state.prediction_result = result
                st.session_state.last_upload_id = file_id
                # Reset per-image states
                st.session_state.current_view_item = result["predicted_food"]
                st.session_state.force_correct_item = None
                st.session_state.show_correction_ui = False
            except Exception as e:
                st.error(f"Lỗi: {e}")
                return

    # Get baseline result
    base_result = st.session_state.prediction_result
    if not base_result:
        return

    # Apply Re-ranking based on Feedback
    top_k = base_result["top_k_items"] # Deduplicated
    reranked_top_k = SessionManager.re_rank(
        top_k, 
        st.session_state.liked_dishes, 
        st.session_state.disliked_dishes
    )
    
    # Determine what to show
    current_food = st.session_state.current_view_item
    predicted_food = base_result["predicted_food"]
    
    # If the user selected a "Correct" item previously, override PREDICTION
    # Wait, "Correct" means "The prediction was WRONG, this IS the actual food."
    # So if they correct, we should update the "Stable Prediction" concept?
    # The requirement says: "A) Stable prediction card (does NOT change unless user uploads a new image)"
    # BUT "If the user selects the correct dish: immediately update the 'current dish' view... and feedback"
    # It does NOT explicitly say update the prediction card label. 
    # However, logically, if I say "This is actually Pho", the card saying "Prediction: Bun Bo" is now known wrong.
    # Requirement 2B: "Current viewing card... changes when user clicks recommendations or corrects".
    # So "Correcting" updates the CURRENT view.
    # The "Stable" card remains as the "System's Initial Guess".
    
    if st.session_state.force_correct_item:
        current_food = st.session_state.force_correct_item
        
    # Get recipe for current view
    if current_food == predicted_food:
        recipe = base_result["recipe"]
        confidence = base_result["confidence"]
    else:
        recipe = recommender.recipe_processor.get_recipe(current_food)
        confidence = None # No confidence for manually browsed items
    
    # Check if personalization is active
    is_personalized = len(st.session_state.liked_dishes) > 0 or len(st.session_state.disliked_dishes) > 0

    # Fetch related (dynamic based on current view)
    related_similar = []
    related_group = []
    if recommender.related_engine:
        related_similar = recommender.related_engine.get_similar_dishes(current_food)
        related_group = recommender.related_engine.get_group_dishes(current_food)
    
    # -------- LAYOUT --------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Ảnh bạn tải lên (không đổi khi bạn duyệt gợi ý)", use_container_width=True)
        
        # --- FEEDBACK PANEL ---
        st.markdown('<div class="feedback-panel">', unsafe_allow_html=True)
        st.write("**Kết quả này thế nào?**")
        
        # Row 1: Correct / Incorrect
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("✅ Chính xác", key="btn_correct"):
                 st.toast("Tuyệt vời! Cảm ơn bạn đã phản hồi.")
                 st.session_state.liked_dishes.add(current_food) 
                 
        with fb_col2:
            if st.button("❌ Sai rồi", key="btn_incorrect"):
                st.session_state.show_correction_ui = True
                
        # Correction UI
        if st.session_state.show_correction_ui:
             st.markdown("---")
             # Create map for dropdown
             display_map = {}
             for item in reranked_top_k:
                 key = item["food_name"]
                 vn_name = get_vietnamese_label(recommender, key)
                 label = f"{vn_name}"
                 display_map[label] = key
                 
             correct_choice_label = st.selectbox(
                "Món đúng là món nào nè?",
                list(display_map.keys()),
                index=None,
                placeholder="Chọn món..."
             )
             
             if correct_choice_label:
                 if st.button("Xác nhận đổi món"):
                     chosen_key = display_map[correct_choice_label]
                     st.session_state.force_correct_item = chosen_key
                     st.session_state.current_view_item = chosen_key
                     st.session_state.liked_dishes.add(chosen_key)
                     st.session_state.show_correction_ui = False
                     st.rerun()

        # Row 2: Like / Dislike
        st.markdown("**Sở thích ăn uống:**")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            if st.button("👍 Thích món này", key="btn_like"):
                st.session_state.liked_dishes.add(current_food)
                if current_food in st.session_state.disliked_dishes:
                    st.session_state.disliked_dishes.remove(current_food)
                st.toast(f"Đã lưu: Bạn thích {current_food}!")
                st.rerun()
        with l_col2:
            if st.button("👎 Không thích", key="btn_dislike"):
                st.session_state.disliked_dishes.add(current_food)
                if current_food in st.session_state.liked_dishes:
                    st.session_state.liked_dishes.remove(current_food)
                st.toast(f"Đã lưu: Bạn không thích {current_food}!")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- TOP K LIST (Reranked) ---
        top_k_header = "### Top dự đoán"
        if is_personalized:
            top_k_header += " (Cá nhân hóa ✨)"
            
        st.markdown(top_k_header)
        for idx, item in enumerate(reranked_top_k[:5]):
            iname = item['food_name']
            iscore = item['score']
            vn_name = get_vietnamese_label(recommender, iname)
            
            marker = ""
            if iname in st.session_state.liked_dishes: marker = "❤️"
            elif iname in st.session_state.disliked_dishes: marker = "🚫"
            
            is_active = (iname == current_food)
            btn_label = f"{marker} {vn_name} ({iscore:.0%})"
            if is_active:
                btn_label = f"📍 {btn_label}"
            
            if st.button(btn_label, key=f"topk_{idx}_{iname}", type="primary" if is_active else "secondary"):
                st.session_state.current_view_item = iname
                st.session_state.show_correction_ui = False
                st.rerun()

    with col2:
        # --- CARD A: SATBLE PREDICTION ---
        pred_vn = get_vietnamese_label(recommender, predicted_food)
        render_stable_prediction_card(
            vn_name=pred_vn, 
            food_key=predicted_food,
            score=base_result["confidence"]
        )

        st.write("") # spacer

        # --- CARD B: CURRENT VIEW ---
        curr_vn = get_vietnamese_label(recommender, current_food)
        render_current_view_card(
            vn_name=curr_vn,
            food_key=current_food
        )
        
        # Return Button
        if current_food != predicted_food:
             if st.button(f"↩ Quay về món nhận diện ({pred_vn})"):
                 st.session_state.current_view_item = predicted_food
                 st.rerun()
        
        if len(st.session_state.liked_dishes) > 0:
             st.caption("💡 Kết quả được cá nhân hóa theo gu của bạn.")

        # Recipe Body
        if recipe:
            # Override title in recipe dict for formatting if needed, 
            # but render_recipe_food_style uses title key which we now load as vietnamese match
            render_recipe_food_style(recipe)
        else:
            st.info("Chưa có thông tin công thức cho món này.")

        # --- RECOMMENDATIONS ---
        st.markdown("---")
        
        # Block 1: Similar
        if related_similar:
            st.markdown('<div class="rec-title">✨ Món tương tự bạn có thể thích</div>', unsafe_allow_html=True)
            r_cols = st.columns(3)
            for idx, item in enumerate(related_similar[:3]):
                with r_cols[idx]:
                    vn = get_vietnamese_label(recommender, item)
                    if st.button(vn, key=f"sim_{idx}_{item}"):
                        st.session_state.current_view_item = item
                        st.session_state.show_correction_ui = False
                        st.rerun()
        
        # Block 2: Group
        if related_group:
            gname = recommender.related_engine.get_group_name(current_food)
            st.markdown(f'<div class="rec-title">📂 Khám phá thêm: {gname}</div>', unsafe_allow_html=True)
            g_cols = st.columns(3)
            for idx, item in enumerate(related_group[:6]):
                if idx % 3 == 0 and idx > 0:
                    g_cols = st.columns(3)
                col_idx = idx % 3
                with g_cols[col_idx]:
                     vn = get_vietnamese_label(recommender, item)
                     if st.button(vn, key=f"grp_{idx}_{item}"):
                        st.session_state.current_view_item = item
                        st.session_state.show_correction_ui = False
                        st.rerun()

if __name__ == "__main__":
    main()
