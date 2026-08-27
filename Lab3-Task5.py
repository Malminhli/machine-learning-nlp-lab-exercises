# Task 5: Evaluating the generated text[cite: 1]
# الغرض من هذا الملف هو توضيح كيفية تقييم النموذج ومراقبة النتائج:
# 1. مراقبة انخفاض قيمة الـ Loss أثناء التدريب للتأكد من تحسن النموذج[cite: 1].
# 2. تجربة جمل بداية مختلفة (start_text)[cite: 1].
# 3. اللعب بمعامل الـ temperature (القيم الأعلى تزيد من عشوائية الحروف، والققيم الأقل تجعل النموذج أكثر تحفظاً وتوقَعاً)[cite: 1].

print("Task 5 Guidelines:")
print("- Test with different start texts.")
print("- Adjust temperature parameter to control randomness of generated text.")