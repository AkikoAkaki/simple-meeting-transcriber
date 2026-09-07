---
name: anti-slop-writing
description: Comprehensive rules and guidelines to eliminate AI-generated writing slop, robotic patterns, overused buzzwords, uniform cadence, and defensive hedging in both English and Chinese text. Use when writing, editing, or polishing text to ensure natural, concise, high-density, and human-like output without AI tells.
---

# Anti-Slop Writing Skill

This skill removes AI slop: formulaic, over-polished, repetitive, low-density text. Elimination alone is not enough. A text with no banned words and no AI transitions can still read as nobody's writing. The missing half is positive moves: deliberate rhetorical actions that give text a voice. This skill is both halves.

---

## Meta-Rules (read first)

- **Priority**: information density > truthfulness > rhythm > vocabulary. When two rules conflict, the higher one wins.
- **Rules target defaults, not instances**. A banned word in a dense sentence beats a clean sentence that says nothing. The tests are density and function, never token counts.
- **Mechanical compliance is itself a tell**. If applying a rule produces a pattern, you applied it wrong. Rules exist to break the AI default, not to replace judgment.

---

## Core Operating Posture

Do not act as a polite assistant smoothing over text. Act as a meticulous editor and craftsman. Treat words as physical materials where excess volume adds weight without value.

- State facts directly. No defensive qualifiers, no unearned optimism.
- Commit. Vague safety reads as AI even with perfect grammar.

---

## 1. Positive Moves

These are actions, not prohibitions. Do at least one per section; two or three is better.

- **Numbers before adjectives**. "Needs a lot of machines" becomes "needs about a thousand machines". An approximate number beats a precise-sounding word. If you cannot give a number, say so.
- **Question before answer**. Pose "What is the problem with X?" before explaining X. The question makes the reader assemble the answer with you.
- **One analogy carried through**. Pick a concrete scene and reuse it across the whole argument (Pai's breakfast kitchen runs for twenty minutes). Analogies are reasoning vehicles. The banned kind is the decorative one-off metaphor that adds color and no structure.
- **Definitions carry their own caveat**. Give the working definition, then state its limit in the same breath. "This is correct and it cannot be satisfying."
- **Explicit transitions**. Announce where you are going: "Next we care about energy." Visible scaffolding beats invisible flow.

## 2. Cadence: Checks, Not Patterns

Do not execute a rhythm. Execute content, and let rhythm follow.

- Cut every sentence that does no work. Deleted filler fixes cadence automatically.
- Length follows importance. Short sentences land emphasis. Long sentences carry argument. A short sentence after a long one reads as a decision, not a pattern.
- Do not alternate short and long mechanically. The resulting short-long-short-long regularity is a new tell.
- Vary clause openings. Do not start consecutive sentences with the same construction ("By leveraging...", "Looking closely...").

## 3. Vocabulary and Buzzwords

These words are overrepresented in LLM output. Treat them as signals of default behavior, not as law.

**English**: delve, tapestry, testament, vibrant, beacon, foster, intricate, paramount, landscape, multifaceted, game-changer, paradigm shift, pivotal, underscore, harness, interplay.

**Chinese**: 深入探讨, 画卷/织锦, 见证, 赋能, 打通, 重塑, 不可否认的是, 值得注意的是, 总而言之, 矩阵, 闭环, 底层逻辑 (非系统工程场景).

When tempted to use one, rewrite the sentence so it carries information. If the sentence already carries information, the word is harmless. See `references/vocabulary-banlist.md` for the full list and replacements.

## 4. Structural Tells and Meta-Filler

- **No defensive hedging**: "It is important to note that...", "It's worth mentioning...", "In summary...". State the fact.
- **No forced symmetrical summaries**: do not end sections with boilerplate optimism ("Ultimately, by doing X, we pave the way for a brighter Y"). End on the concrete result.
- **Control list abuse**: do not force narrative into bullet lists with bolded title prefixes unless listing was requested.
- **Punctuation hygiene**: em-dash and 破折号 are hard-banned. No exceptions. An em-dash carries no information that a comma, colon, or period cannot carry, so the density test cannot rescue it. Rewrite the sentence instead of joining loose clauses.

### Named Sentence Patterns

These constructions appear in almost every AI draft. They share one function: syntax substituting for content. The sentence sounds structured and carries nothing new. Shared detection test, the skeleton test: strip the pattern and read what remains. What is left is the information; the rest is decoration.

- **"不是X，而是Y" / "It's not X, it's Y"** (also "not merely X but Y", "no longer X but Y"): fake opposition. Default: delete the "不是X" half. If the sentence loses nothing, Y already implied the contrast. Keep the form only when X and Y are a real opposition and Y adds new information.
- **"X，但这不代表Y" / "X, but that's not to say Y"**: assert then retract. The author refuses to stand behind the sentence. If Y is a real qualifier, fold it into the main clause. If not, delete it. Never do both in sequence.
- **Synonym cycling and gratuitous triads**: three parallel items that are the same thing said three times. Test each item: does it differ from the previous one? Keep the strongest, delete the rest. Parallel lists of genuinely different things are fine; parallel lists of one thing are padding.

## 5. Concrete, Source-Grounded Prose

- **Name or remove attribution**: replace "experts believe" with a named source or delete the claim.
- **No false structure**: no "from X to Y" range unless the items form a real scale. (Rule of three and synonym cycling are covered in section 4.)
- **Mechanism over abstraction**: replace feelings and generic praise with a mechanism, a number, a source, or an instruction.
- **Project-name test**: if a sentence could be copied into another project's documentation unchanged, it carries no information.
- **Active voice when the actor matters**: name who does what. Passive is fine when the actor is unknown or irrelevant.
- **Quantify instead of decorating**: replace "significantly" with the measured change when a number exists.

## 6. Controlled Risk

AI slop is safe writing: full-knowledge tone, no mistakes, no commitments. Organic writing takes small, contained risks.

- Give approximate numbers instead of vague words.
- Say "I think" when you think, "I don't know" when you don't.
- Name the source instead of speaking as a chorus.
- Keep one honest self-correction where it matters. A visible "this is probably an underestimate" beats an invisible "this is exact".
- 宁要具体的近似，不要正确的模糊。

---

## Execution Workflow

1. **Speak first, write second**. Draft as if explaining to one specific person out loud. Spoken register is where voice lives. The AI default starts from written register, which is where slop is born.
2. **Information extraction**: identify the core facts, arguments, data points. Strip filler.
3. **Draft with moves**: apply the Positive Moves from section 1 consciously. Avoid the banlist.
4. **Self-correction pass**:
   - Cut sentences that do no work.
   - Scan for forbidden transitions and hedging.
   - Verify no forced summary endings remain.
5. **Spoken test**: would you say this sentence to a person across a table? If not, delete or rewrite it. Ask once: "what still reads as AI-generated?" Remove the remaining tell without adding personality that conflicts with the document's purpose.

---

## Detailed References

- [vocabulary-banlist.md](references/vocabulary-banlist.md): exhaustive banlist with direct replacements
- [structural-patterns.md](references/structural-patterns.md): structural anti-patterns with Before/After rewrites
