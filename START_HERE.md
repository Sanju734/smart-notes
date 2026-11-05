# AI Engineering Internship-to-Hire — Take-Home Assessment

## Smart Notes Assistant | Beginner-friendly challenge

**Congratulations on progressing to the technical assessment stage!**

Our internship program offers the opportunity to work on AI-enabled products across **EdTech, SpaceTech, and E-Commerce**. For fair comparison, everyone receives the **same domain-neutral task**. You do not need prior knowledge of any of these industries.

### Time, tools and environment

- **Submission window:** 72 hours. The exact deadline (date, time and timezone) is in your invitation email.
- **Suggested effort:** 6–8 hours of focused work. This is a time *window*, not an instruction to work for 72 hours. Stop when your work is demonstrably functional and documented.
- **Use AI tools freely:** Codex, Claude Code, Cursor, Copilot, ChatGPT, and others are welcome. You will be evaluated on how well you **understand and verify** the result, not on how much code you type manually.
- **No payment needed:** The task can be completed locally with Python and free libraries. No paid LLM API, credit card, GPU, cloud account, or public deployment is required.
- **Submit only synthetic data.** Do not include real personal information, secrets, API keys, or credentials.

### Your task in one sentence

Build a small app that lets a person **save notes, automatically classify them into four categories, and search the notes later**—powered by a text-classification model that **you trained and evaluated**. You will also **measure how well your search works**, and make the app **say when it is unsure** instead of forcing a note that fits no category into one.

Examples:

| Note | Category |
| --- | --- |
| My Python exam starts on Friday. | `Study` |
| Pay the electricity bill by the 28th. | `Money` |
| Flight to Delhi departs on Monday morning. | `Travel` |
| Buy a laptop stand this weekend. | `Shopping` |

### Before you code

Read `TASK.md`, `CATEGORY_GUIDE.md`, and `API_CONTRACT.md`. The `data/` folder contains the supplied training, validation, out-of-scope, and sample notes.

### What to submit

A public GitHub repository (or a source zip) containing the app, the trained model or training instructions, your search test set, tests, a clear `README.md`, `RESULTS.md` and `AI_WORKFLOW.md`. A **2–3 minute demo video** or a few screenshots are welcome but **optional**; they help us review faster and are not scored.

See `SUBMISSION_CHECKLIST.md` and templates. A simple working app with clear reasoning is better than an elaborate unfinished project.

### If you are short on time

Do the required parts in this order: a working `/classify` with a trained model (Parts B and E), the app basics (Part A), then the evaluations (Parts C and D), and tests and docs (Part F). Keep the UI plain; nobody is scoring design. Skip the optional bonus. An honest "I didn't finish X, here's what I would do" in your README is better than a rushed, broken feature.

### What happens next

We run every submission the same way: we follow your README, re-run your validation numbers through your `/classify` endpoint, and test it on notes you have not seen, including notes that fit no category. Shortlisted candidates then have a **20-minute live discussion**, where you run your app, explain your model and make one small change to your code.

### Internship program information

As described in the recruitment announcement, the program is structured as **months 1–2: unpaid internship** focused on learning and practical building; **months 3–6: paid internship for eligible interns**; and **full-time consideration for the top-performing approximately 10% of interns**, based on technical capability, performance and contribution. Completing this assessment or joining the program does **not guarantee** progression to later stages or full-time employment. Refer to the formal internship terms for details.

**The task is about your ability to learn, build, test and explain—not about already knowing every AI framework.**
