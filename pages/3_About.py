"""
About page — project details, methodology, and disclaimer.
"""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ About This Project")

st.markdown(
    """
    ### What this is
    A skin disease image classifier built using transfer learning with
    MobileNetV2, trained to recognize 22 categories of skin conditions
    (plus healthy/normal skin) from photographs.

    ### How it was built
    - **Base model:** MobileNetV2, pretrained on ImageNet
    - **Approach:** Transfer learning in two phases —
        1. Train a new classification head with the base model frozen
        2. Fine-tune the top layers of the base model at a low learning rate
    - **Class imbalance handling:** Class weights computed to counter
      uneven numbers of images per disease category
    - **Data augmentation:** Random rotation, shifts, zoom, and flips
      applied during training to improve generalization

    ### Dataset
    Images organized into train/test folders, one subfolder per disease
    category, covering a range of common dermatological conditions.

    ### Limitations
    - Overall validation accuracy is around 48% across 22 classes
    - Some visually distinctive conditions (like Vitiligo) are classified
      very reliably; others — mostly overlapping inflammatory
      rashes/lesions — are much less reliable
    - This model has **not** been clinically validated and should not be
      used for real medical decisions

    ---

    **Disclaimer:** This tool is for educational and demonstration
    purposes only. It is not a medical device and does not provide medical
    advice, diagnosis, or treatment. Always seek the advice of a qualified
    healthcare provider with any questions regarding a medical condition.
    """
)
