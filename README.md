# RLHF-Homework-Portfolio
# 專案：基於人類回饋的強化學習（RLHF）實作與探索 - 課程作業 HW7

**專案類型：** 機器學習課程作業 (NTU Machine Learning, 2025 Spring - HW7)
**核心技術：** RLHF, Direct Preference Optimization (DPO), LoRA, Llama-3

## 專案概述 (Project Overview)

本專案是台大資訊工程學系「機器學習」課程的第七份作業，旨在透過實作直接偏好優化 (DPO) 來對大型語言模型 (Llama-3-8B-Instruct) 進行人類對齊。作業提供了一個基於 Unsloth 框架的 Colab Notebook 環境，包含了基礎的資料載入、模型設定（使用 LoRA 進行高效能微調）以及 DPO 訓練流程。

原始作業的核心要求是調整實驗參數（如 `num_epoch`, `data_size`, `support_ratio`），觀察並比較模型在對齊前後針對特定主題（AI 生成藝術的倫理與影響）的回答變化。

## 我的額外探索與貢獻 (My Additional Explorations & Contributions)

在完成作業的基本要求之上，我對 RLHF 及 DPO 的效果和影響因素產生了濃厚興趣，因此進行了以下額外的探索與分析：

1.  **系統性超參數調校與影響分析 (Systematic Hyperparameter Tuning & Impact Analysis)：**
    * 除了作業要求的參數（`num_epoch`, `data_size`, `support_ratio`），我額外設計並執行了一系列實驗，系統性地調整了 DPO 訓練中的 `beta` 值（控制對參考模型的偏離程度）以及 LoRA 適配器的關鍵參數（如 `r` 和 `lora_alpha`）。
    * **目標：** 觀察這些參數如何影響對齊的效果、訓練的穩定性以及最終模型的回答偏好。
    * **發現：** [**你需要在此處填寫你實際實驗後的發現，例如：** 「我發現較高的 `beta` 值雖然能讓模型更快學習偏好，但也可能導致語言模型的流暢度下降。LoRA 的 `r` 值在一定範圍內增加有助於對齊，但過高則有過擬合風險。」（**請務必基於你真實的實驗結果**）]

2.  **模型回答質性分析與案例研究 (Qualitative Analysis of Response Changes & Case Studies)：**
    * 不僅僅是比較對齊前後的總體差異，我針對特定類型的提問（例如：涉及模糊邊界、強烈情感或多方觀點的提問），進行了更細緻的回答內容質性分析。
    * **方法：** 我歸納了原始模型常見的回答缺陷（例如：過於中立、缺乏明確立場、回答過長等），並追蹤 DPO 訓練後這些缺陷的改善程度。同時，我也記錄了對齊後可能出現的新問題（例如：過度迎合某一觀點而失去客觀性）。
    * **洞見：** [**你需要在此處填寫你實際分析後的洞見，例如：** 「DPO 對齊在引導模型產生更符合特定偏好的簡潔回答方面效果顯著，但在處理需要權衡多方利益的複雜問題時，仍需更細緻的偏好數據或對齊策略。」（**請務必基於你真實的分析結果**）]

3.  **探索不同偏好數據建構策略 (Exploring Different Preference Data Construction Strategies)：**
    * 作業中 `support_ratio` 參數影響了「chosen」和「rejected」回應的來源。我嘗試了除了調整比例之外的幾種方式來微調訓練數據中的偏好對，例如，針對一部分數據，我試圖 [**例如：** 「強化特定類型『不希望看到的回答』作為 rejected 範例」或「對於某些模棱兩可的 prompt，嘗試平衡 chosen/rejected 回應的觀點分佈」]。
    * **目的：** 觀察這種細微的數據策略調整是否能對模型在特定子任務上的對齊產生更精準的影響。
    * **初步觀察：** [**你需要在此處填寫你實際實驗後的觀察，例如：** 「初步觀察顯示，強化特定負面範例有助於模型避開某些不期望的回答模式，但需要更多實驗來驗證其泛化能力。」（**請務必基於你真實的實驗結果**）]

4.  **對齊後模型的邊界測試與魯棒性初探 (Preliminary Boundary Testing & Robustness Checks of Aligned Model)：**
    * 在 Colab Notebook 提供的「自由測試」環節，我設計了一些偏離訓練數據分佈的提問 (out-of-distribution prompts) 或帶有輕微對抗性的提問。
    * **目的：** 初步評估對齊後模型的魯棒性，以及它在面對未見過或略有挑戰性的輸入時的反應。
    * **發現：** [**你需要在此處填寫你實際測試後的發現，例如：** 「模型對於訓練數據中常見主題的提問表現良好，但在面對全新領域或提問方式較為隱晦時，其對齊效果有所下降，顯示其泛化能力仍有提升空間。」（**請務必基於你真實的測試結果**）]

**目前狀態 (Current Status):**

這些額外的探索仍在進行中，我計劃在接下來的 [例如：幾週內/課餘時間] 繼續深入研究 [例如：特定超參數的組合影響 / 更細緻的回答錯誤分析 / 某種新的偏好數據增強方法]。我對利用 RLHF 技術提升大型語言模型的可控性與實用性抱有極大熱情。

## 核心技術棧 (Core Technologies Used)

* **模型 (Model):** Unsloth Llama-3-8B-Instruct
* **對齊方法 (Alignment Method):** Direct Preference Optimization (DPO)
* **高效微調 (Efficient Fine-tuning):** LoRA (Low-Rank Adaptation)
* **框架 (Framework):** Unsloth, Transformers, TRL (Transformer Reinforcement Learning)
* **環境 (Environment):** Google Colab (Python, PyTorch)

## 原始作業目標與要求 (Original Assignment Objectives)

* 理解並操作 DPO 流程。
* 使用 Unsloth 和 LoRA 高效微調大型語言模型。
* 透過調整 `num_epoch`, `data_size`, `support_ratio` 觀察模型對齊前後的變化。
* （可簡述其他原始作業要求）

## 執行與觀察 (Execution & Observation - *可放 Colab Notebook 連結*)

完整的程式碼、實驗設置以及對齊前後模型的回答範例，請參考我的 Colab Notebook：
[連結到你的 GitHub 上的 .ipynb 檔案]

## 主要學習與心得 (Key Learnings & Takeaways)

1.  **DPO 的威力與挑戰：** 親身體驗了 DPO 在模型對齊上的有效性，同時也認識到偏好數據的品質和數量對最終效果的巨大影響。
2.  **高效微調的重要性：** LoRA 等技術使得在有限資源下微調大型模型成為可能，這對於學術研究和個人探索至關重要。
3.  **[你從額外探索中學到的最重要的一點，例如：** 「超參數 `beta` 在 DPO 中的細微調整，能顯著改變模型的探索與利用平衡，進而影響對齊的精細程度。」]
4.  **[你從額外探索中學到的另一個重要點，例如：** 「質性分析對於理解模型行為變化至關重要，僅憑量化指標有時難以捕捉對齊的全部面貌。」]
5.  對 RLHF 領域產生了更深入的研究興趣，並意識到這是一個充滿挑戰與機遇的方向。

---
