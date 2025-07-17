# Japanese Sentiment Analysis Annotation Guidelines
# 日本語感情分析アノテーションガイドライン

Version 2.3 | Last Updated: 2024-12-15

## 1. Overview / 概要

This document provides comprehensive guidelines for annotating Japanese text for sentiment analysis. The goal is to achieve consistent, high-quality annotations with inter-annotator agreement (IAA) > 0.90.

本ガイドラインは、日本語テキストの感情分析アノテーションのための包括的な指針を提供します。目標は、アノテーター間一致率（IAA）0.90以上の一貫した高品質なアノテーションの実現です。

## 2. Sentiment Categories / 感情カテゴリー

### 2.1 Primary Categories / 主要カテゴリー

| Label | Japanese | Description | Example |
|-------|----------|-------------|---------|
| POS | ポジティブ | Positive sentiment | 素晴らしい商品でした！期待以上です。 |
| NEG | ネガティブ | Negative sentiment | 全く使い物にならない。がっかりしました。 |
| NEU | 中立 | Neutral/Mixed sentiment | 普通の商品です。特に問題はありません。 |

### 2.2 Fine-grained Categories / 詳細カテゴリー

| Label | Emotion | Japanese | Confidence Required |
|-------|---------|----------|-------------------|
| JOY | Joy | 喜び | High (>0.8) |
| TRUST | Trust | 信頼 | Medium (>0.6) |
| FEAR | Fear | 恐れ | Medium (>0.6) |
| SURPRISE | Surprise | 驚き | High (>0.8) |
| SADNESS | Sadness | 悲しみ | High (>0.8) |
| DISGUST | Disgust | 嫌悪 | Medium (>0.6) |
| ANGER | Anger | 怒り | High (>0.8) |
| ANTICIPATION | Anticipation | 期待 | Low (>0.4) |

## 3. Annotation Process / アノテーションプロセス

### 3.1 Step-by-Step Procedure / 段階的手順

1. **Initial Reading / 初読**
   - Read the entire text without judgment
   - テキスト全体を先入観なく読む

2. **Context Analysis / 文脈分析**
   - Identify speaker's intent
   - Consider cultural context
   - 話者の意図を特定
   - 文化的文脈を考慮

3. **Primary Labeling / 一次ラベリング**
   - Assign POS/NEG/NEU label
   - Record confidence (0-1)
   - POS/NEG/NEUラベルを付与
   - 確信度（0-1）を記録

4. **Fine-grained Labeling / 詳細ラベリング**
   - If confidence > 0.7, add emotion labels
   - 確信度 > 0.7の場合、感情ラベルを追加

5. **Quality Check / 品質確認**
   - Review for consistency
   - Check edge cases against guidelines
   - 一貫性を確認
   - エッジケースをガイドラインと照合

### 3.2 Decision Tree / 判定フローチャート

```
[Text Input]
    |
    v
[Contains explicit emotion words?]
    |           |
   Yes          No
    |           |
    v           v
[Check polarity] [Analyze context]
    |               |
    v               v
[Assign label]  [Check implicit cues]
    |               |
    v               v
[Confidence?]   [Assign NEU if unclear]
    |
    v
[Add fine-grained if >0.7]
```

## 4. Special Cases / 特殊ケース

### 4.1 Japanese-Specific Considerations / 日本語特有の考慮事項

#### A. Keigo (Honorifics) / 敬語
- Polite criticism may still be NEG
- 丁寧な批判もNEGとする

Example:
```
❌ "恐れ入りますが、品質に問題があるように思われます" → NEU
✅ "恐れ入りますが、品質に問題があるように思われます" → NEG
```

#### B. Indirect Expression / 間接表現
- Japanese often uses indirect criticism
- 日本語では間接的な批判が多い

Example:
```
"ちょっと..." → Usually NEG (context-dependent)
"微妙ですね" → Usually NEG or NEU
```

#### C. Sarcasm & Irony / 皮肉・アイロニー
- Look for contradiction between form and content
- 形式と内容の矛盾を確認

Example:
```
"素晴らしいですね（笑）" → NEG (if context suggests sarcasm)
```

### 4.2 Multi-sentence Handling / 複数文の処理

1. **Overall Sentiment Priority / 全体感情を優先**
   - Final sentence often carries main sentiment
   - 最終文が主要な感情を持つことが多い

2. **Aspect-based Conflicts / アスペクトベースの対立**
   ```
   "料理は美味しかったが、サービスが最悪だった。二度と行かない。"
   Primary: NEG (due to conclusion)
   Aspects: Food=POS, Service=NEG, Overall=NEG
   ```

## 5. Quality Assurance / 品質保証

### 5.1 Self-Check Criteria / セルフチェック基準

- [ ] Is the annotation consistent with similar examples?
- [ ] Have I considered cultural context?
- [ ] Is my confidence score honest?
- [ ] Would another annotator likely agree?

### 5.2 Common Errors / よくある間違い

| Error Type | Example | Correction |
|------------|---------|------------|
| Over-interpreting politeness | "ありがとうございました" → POS | Should be NEU unless context indicates genuine gratitude |
| Missing sarcasm | "最高ですね（皮肉）" → POS | Should be NEG |
| Ignoring conclusion | Mixed review ending negatively → NEU | Should be NEG |

## 6. Inter-Annotator Disagreement Resolution / アノテーター間不一致の解決

### 6.1 Discussion Protocol / 議論プロトコル

1. **Identify Disagreement Type**
   - Ambiguity-based
   - Interpretation-based
   - Guideline clarity issue

2. **Evidence Presentation**
   - Each annotator presents reasoning
   - Reference specific guideline sections

3. **Consensus Building**
   - If no consensus, consult expert annotator
   - Document edge case for guideline update

### 6.2 Arbitration Rules / 仲裁ルール

- 3+ annotators: Majority vote
- 2 annotators: Expert arbitration
- Document all arbitrated cases

## 7. Tool-Specific Instructions / ツール固有の指示

### 7.1 Web Interface Usage

```python
# Example annotation format
{
    "text_id": "sent_001",
    "text": "この商品は期待以上でした！",
    "primary_label": "POS",
    "confidence": 0.95,
    "fine_grained": ["JOY", "SURPRISE"],
    "annotator_id": "A001",
    "timestamp": "2024-12-15T10:30:00Z",
    "notes": "Explicit positive expression with surprise element"
}
```

### 7.2 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1 | Label as POS |
| 2 | Label as NEG |
| 3 | Label as NEU |
| Space | Next sample |
| B | Previous sample |
| F | Flag for review |

## 8. Metrics and Evaluation / 評価指標

### 8.1 Required Metrics

1. **Cohen's Kappa (κ)**: Target > 0.85
2. **Krippendorff's Alpha (α)**: Target > 0.80
3. **Percentage Agreement**: Target > 90%

### 8.2 Performance Tracking

```python
# Calculate your annotation metrics
from jmmaf.metrics import AnnotationMetrics

metrics = AnnotationMetrics()
report = metrics.calculate_agreement(
    annotator_1_labels,
    annotator_2_labels
)
print(f"Kappa: {report['kappa']:.3f}")
print(f"Alpha: {report['alpha']:.3f}")
```

## 9. Continuous Improvement / 継続的改善

### 9.1 Weekly Calibration Sessions

- Review disagreement cases
- Update guidelines based on edge cases
- Practice on new domain data

### 9.2 Feedback Loop

1. Annotate 100 samples
2. Calculate agreement with gold standard
3. Review errors
4. Adjust approach
5. Repeat

## 10. Appendix / 付録

### A. Emotion Word Dictionary / 感情語辞書

[Partial list - full dictionary available in `/resources/emotion_dictionary.json`]

| Word | Reading | Category | Strength |
|------|---------|----------|----------|
| 嬉しい | うれしい | JOY | 0.8 |
| 悲しい | かなしい | SADNESS | 0.9 |
| 怖い | こわい | FEAR | 0.7 |
| 驚く | おどろく | SURPRISE | 0.8 |

### B. Domain-Specific Adjustments / ドメイン固有の調整

- **Product Reviews**: Focus on overall recommendation
- **Social Media**: Consider emoji and context
- **News Comments**: Account for political bias
- **Customer Service**: Distinguish complaint severity

### C. Change Log

- v2.3 (2024-12-15): Added multi-modal annotation support
- v2.2 (2024-11-01): Enhanced sarcasm detection guidelines
- v2.1 (2024-09-15): Added fine-grained emotion categories
- v2.0 (2024-07-01): Major revision based on 10k annotation analysis

---

**Questions?** Contact lead annotator at annotations@jmmaf-project.org

**最終更新者**: 柳澤 亮 (Ryo Yanagisawa)