# Video script - Assignment 3 (whole assignment, final)

**YouTube title:** `CO3133-Semester-261 Group G-roundhog Assignment 3`
**Visibility:** Unlisted or Public.

**Length and split:** about 12 minutes in total.
- **[T]** Phan Bá Thanh: problem, data, models and experiment design, slides 0-7, about 5.5 min.
- **[P]** Trần Công Hoàng Phước: results, analysis and conclusion, slides 8-18, about 6.5 min.

**How to use this script.**
- **Quoted text** is spoken word for word, at a calm pace of about 130-140 words per minute.
- ***[Italics in brackets]*** are stage directions (what to point at or click). They are not spoken.
- **Bold numbers** must be said exactly as written.
- Rehearse each block two or three times, until you can say it while looking at the slide instead of the paper. The course asks presenters to explain and defend the work, not to read the slides.

**Slide numbers** follow the page footer of `docs/a3-slides/slides.pdf`: the title slide is slide 0 and the last slide is 18/18. The four section dividers have no number: just click through them.

---

## [T] 0:00-0:30 · Slide 0: title

*[Camera on, title slide visible.]*

"Hello everyone. We are group G-roundhog, CO3133, semester 261. I am Phan Bá Thanh, and with me is Trần Công Hoàng Phước. Our Assignment 3 is multimodal classification. A recipe web page gives us one image and its page title, and we predict which of **101** dishes the page is about. I will present the problem, the data and the models. Then Phước will present the results and the analysis."

## [T] 0:30-0:50 · Slide 1: outline

*[Next slide.]*

"The talk has four parts. First, the data, where most titles name the class. Second, the models, including our improved fusion, F2. Third, the results: F2 beats the simple trained fusion, ties the late fusion, and wins when we hide the class names. Fourth, an analysis of the cases where the two modalities disagree."

## [T] 0:50-1:35 · Slide 2: task

*[Click through the section divider, then the next slide. Point at a few pairs.]*

"The dataset is UPMC Food-101. Every item is one image found on a recipe page, plus the title of that page. Image and title come from the same page; they are never paired at random. After our validity checks we have about **62,800** training pairs, **4,800** validation pairs and **22,594** test pairs.

The pairs are noisy. In a manual audit of **200** pairs, **81.5 %** were fully valid. When one side is bad, it is usually only one side: a logo instead of food, or a title like 'Homemade'. Wrong labels were rare, **0.5 %**."

## [T] 1:35-2:25 · Slide 3: two leaks

*[Next slide. Point at the first row, then at the last row.]*

"Before training anything, we found two leaks that shape everything we report.

First, the pages were collected with searches like 'apple pie recipe', so **79 %** of the titles contain the class name. A simple keyword rule already reaches **77 %** accuracy.

Second, the official train and test lists share items. **39 %** of the test titles also appear in training, and **14.5 %** of the test images have a near-duplicate in training.

*[Point at 'Control'.]* So every result we show is also checked on two extra test views: titles never seen in training, and images without duplicates. And in experiment E2 we mask the class names completely."

## [T] 2:25-3:10 · Slide 4: models

*[Click through the section divider, then the next slide.]*

"Here are the models. UA is a ResNet-50 that only sees the image. UB is DistilBERT, which only sees the title. K0 and T0 are cheap text references: the keyword rule, and TF-IDF with logistic regression.

Then three fusions. F0 is a late fusion: a weighted average of the two softmax outputs, with the weight tuned on validation. F1 concatenates the two pooled feature vectors and trains everything end to end. *[Point at F2.]* F2 is our improved fusion. Every trained model uses the same split and three seeds, and we compare models with the decision rule from our proposal."

## [T] 3:10-3:55 · Slide 5: why a new fusion

*[Next slide.]*

"Why did we need F2? In M2, the trained fusion F1 lost to the simple average F0, and this table shows why. F1 fits the training set to **99.7 %**, because the title alone is enough to memorise the labels. So the image branch stops learning. *[Point at 53.3.]* If we remove the title at test time, F1 drops to **53.3 %**, far below the image-only model at **70.7 %**. And when only the image is right, F1 still follows the wrong title in more than a quarter of the items. This is called modality laziness."

## [T] 3:55-4:55 · Slide 6: F2

*[Next slide. Follow the arrows from left to right.]*

"F2 attacks this problem in three ways.

First, cross-modal attention. The ResNet gives **49** region features, and DistilBERT gives one feature per word. All of them go, together with a learned fusion token, into two transformer layers. So every image region can look at every word, and the fusion token collects the evidence for the final decision.

*[Point at the dashed heads.]* Second, auxiliary losses. Each encoder has its own small classification head, and its loss is added with weight **0.5**. So the image encoder must stay useful on its own.

*[Point at the box at the bottom.]* Third, modality dropout. For **15 %** of the training items we remove the image or the title and put a learned 'missing' token in its place. The model learns to work with one modality alone.

F2 has **98 million** parameters, only **7 %** more computation than F1."

## [T] 4:55-5:30 · Slide 7: controlled experiments

*[Next slide.]*

"To test each idea separately, we changed one factor at a time and kept the split and the seeds. E1 removes the auxiliary losses. E4 removes the modality dropout. E2 masks every class name in every title and retrains the text models and all fusions. E-mod removes the title or the image from part of the test set. That is **18** new training runs, about **5.7** GPU-hours on a lab RTX A6000. Phước will now show what we found."

---

## [P] 5:30-6:05 · Slide 8: main result

*[Click through the 'Results' divider, then the next slide.]*

"Thank you, Thanh. This is the main result, in test macro-F1 averaged over three seeds. The image model reaches **70.8**, the text model **86.8**. Interestingly, the simple TF-IDF model is just as good as DistilBERT on these short titles. *[Point at the three right bars.]* All three fusions are around **93** to **94**, so combining the two modalities adds six to seven points. This holds on the full test set, without duplicate images, and on titles never seen in training."

## [P] 6:05-6:45 · Slide 9: decision rule

*[Next slide.]*

"Now the comparisons, with our decision rule. A model wins only if the gap is larger than the seed spread, and McNemar's test gives p below **0.05**. *[Point at the third row.]* F2 beats F1 by **0.73** points, which is ten times the seed spread. *[Point at the last row.]* But F2 does not beat the late fusion F0: the gap is **0.33** points and McNemar gives **p = 0.5**. With two strong unimodal models and informative titles, a simple average is a very hard baseline. One more point for F2: it is the best-calibrated model we have."

## [P] 6:45-7:25 · Slide 10: E1, auxiliary losses

*[Next slide.]*

"Do the auxiliary losses do their job? Look at the middle column. *[Point at 69.7.]* In F2, the image head alone reaches **69.7 %**, as good as the separate ResNet-50. Without the auxiliary losses, a probe on the same image features only gets **62.3 %**. So without them the image encoder gets lazy, exactly as in F1. The overall macro-F1 also drops by **0.46** points, which is significant. The biggest effect is when the title is missing: **65.3** against **58.6**."

## [P] 7:25-8:10 · Slide 11: missing title

*[Next slide. Point at the right end of the curves.]*

"This plot removes the title from more and more test items. The interesting values are on the right, with no titles at all. F1 falls to **52.5**. Our F2 keeps **65.3**. Without modality dropout it would be **60.1**; without the auxiliary losses, **58.5**. So both ideas help by five to seven points, and modality dropout costs nothing on complete pairs. *[Point at the green line.]* Still, F0 is the most robust, because it simply falls back to the image model, at **70.8**."

## [P] 8:10-9:00 · Slide 12: E2, class names masked

*[Next slide.]*

"This is our most important experiment. We replaced every class name in every title with a mask and retrained. *[Point at 43.7.]* The text model falls from **86.8** to **43.7**: half of the text signal was just the dish name.

Now look at the fusions. On masked titles, F2 is the best, with **81.3**, which is **2.5** points above the late fusion and almost one point above F1. Both differences are significant. When the title no longer gives the answer directly, the cross-attention of F2 combines partial evidence better than averaging two predictions. So the ranking between F0 and F2 depends on how informative the text is."

## [P] 9:00-9:45 · Slide 13: support and conflict

*[Click through the 'Analysis' divider, then the next slide. Point at 'conflict_I' in the left panel.]*

"Where do the fusions gain? We split the test items by which unimodal model is right. On items where both are right, every fusion is right too. The gain comes from the conflicts. When only the image is right, so the title is misleading, F2 recovers **75 %**, against **69 %** for F1, but F0 still gets **91 %**. *[Point at the right panel.]* With masked titles the picture changes: these conflicts become the biggest group, and F2 leads on the items where only the text is right."

## [P] 9:45-10:20 · Slide 14: inside F2

*[Next slide. Point at the titles above the images.]*

"Here we look inside F2. These are items where the title is useless: 'Account Suspended', 'Hello', 'In Rome'. The fusion token gives these titles almost no attention, between zero and four percent, and decides from the image. The heat maps show that the image encoder looks at the food. On average, F2 attends to the image more on these items than on the others, so it adapts item by item."

## [P] 10:20-11:05 · Slide 15: why the unimodal models fail

*[Next slide.]*

"We also labelled **100** random conflict items by hand, to find out why one modality fails. When the text model is wrong, the title is usually generic, like a site name. F2 fixes **93 %** of those. *[Point at the orange numbers.]* The hard case is a title that names another dish, for example 'French Bread Pizza' on a pizza photo, which the text model reads as garlic bread. Every fusion gets only half of these right. When the image model is wrong, the dish usually looks like another dish, and F2 fixes all of them. Wrong labels were only **2** of the **100** items."

## [P] 11:05-11:20 · Slide 16: cost

*[Next slide.]*

"The improvement is cheap. F2 has **6 million** more parameters than F1, **7 %** more computation, the same training time of about **19** minutes per run, and **8.5** milliseconds per item."

## [P] 11:20-11:50 · Slide 17: conclusions

*[Next slide.]*

"To conclude. Fusing the image and the title adds six to seven points over the best single modality. The title is a strong but leaky signal, so we always check masked and unseen titles. F2 beats the simple trained fusion, keeps its image branch alive and copes better with a missing title. It ties the late fusion on normal titles, but wins by two and a half points when the class names are hidden. So in practice: use the late fusion when titles are informative, and F2 when the text is weak or you need calibrated confidence."

## [P] 11:50-12:00 · Slide 18: links

*[Last slide.]*

"The code, the report and all checkpoints are linked here. Thank you for watching."

---

## Recording checklist

**Before recording**
- Record in English with screen share of `slides.pdf` in full screen. Camera on is optional.
- Each speaker records their own part; join the two parts in a video editor.
- Check that the footer number on screen matches the script.

**After uploading**
- Put the YouTube link into three places:
  - `docs/a3-report/sections/declarations.tex` (`[video URL]`);
  - `README.md`;
  - `docs/REPORT_M3.md`.
- Then rebuild the PDF report: `cd docs/a3-report && latexmk -pdf main.tex`.
