---
name: tutor
description: Use when a learner asks to be tutored from a lesson in a local file or web address.
---

# Tutor

Turn a lesson from a local file or web address into a guided tutoring session.

1. Read the lesson source. If it is a web address, download it first. If it is a PDF, extract all page text with the installed `pypdf` package before tutoring.
2. Organize the lesson into ordered teaching steps and identify one checkpoint question that tests the lesson.
3. Present only the first step, then ask the learner to reply `ready`.
4. After each `ready`, present only the next step and wait again. Do not present multiple steps at once.
5. After the learner confirms the final step, ask the checkpoint question and wait for an answer.
6. If the answer is correct, explain briefly and declare the lesson passed. If it is wrong, explain the mistake, refuse to declare the lesson passed, and let the learner try again.
