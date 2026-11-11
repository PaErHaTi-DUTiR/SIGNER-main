# CA-STNER — Annotation Guidelines

**Dataset:** CA-STNER (Central Asian Speech-Text Named Entity Recognition)  
**Languages:** Kazakh (kk) · Kyrgyz (ky) · Tajik (tg) · Turkmen (tk) · Uzbek (uz)  
**Entity Types:** PER · ORG · LOC  
**Annotation Task:** Review and correct LLM-generated NER labels on speech transcripts

---

## 1. Overview

CA-STNER is a Named Entity Recognition dataset built from speech recordings in five Central Asian languages. Each source instance pairs audio with an existing crowd-verified source transcript, not a new ASR transcription. Annotators identify **person names (PER)**, **organizations (ORG)**, and **locations (LOC)** in the transcript, consulting audio when needed for disambiguation.

**Annotation mode:** Reviewers are presented with LLM pre-annotations (shown as colored highlights). Your task is to:
1. **Confirm** correct pre-annotations (leave unchanged)
2. **Correct** wrong labels or boundaries (modify the span or label)
3. **Delete** spurious entities (remove hallucinated labels)
4. **Add** missed entities (draw new spans)

---

## 2. General Principles

### 2.1 Minimal Span
Annotate the **shortest surface form** that identifies the entity. Do not include surrounding context words, articles, or descriptors unless they form part of the proper name.

| ✓ Correct | ✗ Incorrect | Note |
|-----------|-------------|------|
| `Алматы` [LOC] | `Алматы қаласы` [LOC] | "қаласы" = "city of", not part of name |
| `Назарбаев` [PER] | `Президент Назарбаев` [PER] | Titles are excluded |
| `БҰҰ` [ORG] | `халықаралық ұйым БҰҰ` [ORG] | Descriptors excluded |

### 2.2 Annotate Surface Form Including Morphological Suffixes
Central Asian languages attach **grammatical suffixes** directly to proper nouns (case, possession, plural). Annotate the **full inflected form** as it appears in the text — do not strip suffixes.

| Language | Example | Annotation | Gloss |
|----------|---------|------------|-------|
| Kazakh (kk) | `Қазақстанда` | `Қазақстанда` [LOC] | in Kazakhstan |
| Kazakh (kk) | `Назарбаевтың` | `Назарбаевтың` [PER] | of Nazarbayev |
| Kyrgyz (ky) | `Бишкекке` | `Бишкекке` [LOC] | to Bishkek |
| Tajik (tg) | `Душанбе` | `Душанбе` [LOC] | Dushanbe (uninflected) |
| Turkmen (tk) | `Aşgabatda` | `Aşgabatda` [LOC] | in Ashgabat |
| Uzbek (uz) | `Toshkentda` | `Toshkentda` [LOC] | in Tashkent |

### 2.3 Multi-Token Entities
Include all tokens that form a single named entity as one contiguous span.

> `Жақыпбек Бекжанов` [PER] *(both first and last name)*  
> `Орта Азия` [LOC] *(full multi-word toponym)*  
> `Дүниежүзілік денсаулық сақтау ұйымы` [ORG] *(full organization name)*

### 2.4 Nested Entities
Do **not** create nested annotations. If a larger phrase contains a named entity, label only the inner proper name.

> `Нью-Йорк университеті` → label `Нью-Йорк` [LOC] and `Нью-Йорк университеті` [ORG] as two separate entities only if both are individually referenced. Otherwise, label only the most specific entity that fits.

### 2.5 When in Doubt, Do Not Label
If you are uncertain whether a mention refers to a real named entity (vs. a common noun or generic reference), **leave it unlabeled**. Precision is preferred over recall.

---

## 3. Entity Type Definitions

### 3.1 PER — Person Name

**Definition:** A PER entity is the proper name (or pseudonym/alias) of a real, historical, or clearly fictional human individual. It includes full names, partial names (surname-only or given-name-only when unambiguous), and transliterated names from other languages.

**Include:**
- Real individuals: politicians, athletes, scientists, artists, historical figures
- Fictional characters treated as individuals in context
- Name transliterations from Russian, Arabic, Chinese, English, etc.
- Surnames used alone when clearly referring to a specific person
- Patronymics and full CIS-style names (given name + patronymic + surname)

**Exclude:**
- Titles, honorifics, or occupations attached to names (*президент*, *академик*, *доктор*)
- Generic references to humans (*адам*, *кісі*, *шахс*, *adam*)
- Ethnic or national group nouns used generically (see ORG section for group entities)
- Pronouns

**Examples:**

| Language | Text (excerpt) | Annotation |
|----------|----------------|------------|
| kk | `Нұрсұлтан Назарбаев президент болды` | `Нұрсұлтан Назарбаев` [PER] |
| kk | `Абайдың поэзиясы` | `Абайдың` [PER] |
| ky | `Чыңгыз Айтматов жазган` | `Чыңгыз Айтматов` [PER] |
| tg | `Эмомалӣ Раҳмон гуфт` | `Эмомалӣ Раҳмон` [PER] |
| tk | `Berdimuhamedow aýtdy` | `Berdimuhamedow` [PER] |
| uz | `Alisher Navoiy she'rlari` | `Alisher Navoiy` [PER] |

**Edge Cases:**

| Case | Decision | Rationale |
|------|----------|-----------|
| `Абай` (poet) without context | PER if clearly a person reference | Unambiguous historical figure |
| `Алла` (God) in religious context | ✗ Not PER | Refers to deity, not a named person |
| `Александр Македонский` | PER (full epithet-name) | Historical proper name |
| Nickname/alias (`Ленин`) | PER | Pseudonyms are PER |

---

### 3.2 ORG — Organization

**Definition:** An ORG entity is the proper name of any formally constituted group of people organized for a specific purpose, including governments, political bodies, companies, international bodies, media outlets, educational institutions, sports teams, and ethnic or national groups when referenced as a collective entity.

**Include:**
- Government ministries, agencies, parliaments
- International organizations (UN, WHO, NATO, SCO, etc.)
- Companies, corporations, state enterprises
- Political parties
- Universities and research institutes
- News agencies and media outlets
- Sports teams and clubs
- Ethnic, national, or religious groups when named as a collective (e.g., *қазақтар* "the Kazakhs" as a people)
- Abbreviations/acronyms of organizations (`БҰҰ`, `ШЫҰ`, `NATO`, `ООН`)

**Exclude:**
- Informal or ad hoc groups with no proper name
- Generic references (*компания*, *ұйым*, *hukumat* without a specific name)
- Government departments described only by function without a proper name

**Examples:**

| Language | Text (excerpt) | Annotation |
|----------|----------------|------------|
| kk | `Біріккен Ұлттар Ұйымы мәжілісі` | `Біріккен Ұлттар Ұйымы` [ORG] |
| kk | `Қазақстан Республикасының Үкіметі` | `Қазақстан Республикасының Үкіметі` [ORG] |
| ky | `Жогорку Кеңеш чечим кабыл алды` | `Жогорку Кеңеш` [ORG] |
| tg | `Созмони Миллали Муттаҳид` | `Созмони Миллали Муттаҳид` [ORG] |
| tk | `Birleşen Milletler Guramasy` | `Birleşen Milletler Guramasy` [ORG] |
| uz | `BMT Bosh Assambleyasi` | `BMT` [ORG] |

**Edge Cases:**

| Case | Decision | Rationale |
|------|----------|-----------|
| `қазақтар` (Kazakhs as a people) | ORG if referring to the group as a collective agent | Ethnic group acting collectively |
| `мұсылмандар` (Muslims) | ORG if referenced as a religious community | Religious collective |
| `Қазақстан Үкіметі` | ORG | Government body with proper name |
| `ол үкімет` (that government) | ✗ Not ORG | Pronoun + generic noun, no proper name |

---

### 3.3 LOC — Location

**Definition:** A LOC entity is the proper name of any geographical or spatial location, including political-administrative entities (countries, cities, regions), natural features (rivers, mountains, seas), man-made structures (streets, buildings, stadiums), and named celestial bodies.

**Include:**
- Countries, states, provinces, districts, cities, villages
- Continents and regions (`Орта Азия`, `Шығыс Еуропа`)
- Rivers, lakes, mountains, deserts, seas, oceans
- Streets, squares, neighborhoods
- Named buildings, airports, stadiums, universities (when referenced as a place)
- Geopolitical or cultural regions (`Таяу Шығыс`, `Орта Шығыс`)

**Exclude:**
- Generic spatial nouns without a proper name (*өңір* "region", *қала* "city", *дарё* "river" alone)
- Directional references (*солтүстікте* "in the north" without a place name)
- Nationalities or demonyms (these are PER or ORG depending on context)

**Examples:**

| Language | Text (excerpt) | Annotation |
|----------|----------------|------------|
| kk | `Алматыда өткен съезд` | `Алматыда` [LOC] |
| kk | `Каспий теңізі жағасында` | `Каспий теңізі` [LOC] |
| ky | `Кыргызстандын борбору Бишкек` | `Кыргызстандын` [LOC], `Бишкек` [LOC] |
| tg | `дар шаҳри Душанбе` | `Душанбе` [LOC] |
| tk | `Türkmenistanda geçirilen` | `Türkmenistanda` [LOC] |
| uz | `O'zbekiston Respublikasi` | `O'zbekiston Respublikasi` [LOC] |

**Edge Cases:**

| Case | Decision | Rationale |
|------|----------|-----------|
| `Қазақстан Республикасы` | LOC (country name) | Political-administrative entity |
| `Қазақстан Республикасының Үкіметі` | ORG | Organization, not the country itself |
| `Ресей` (Russia) | LOC | Country name |
| `ресейлік` (Russian adj.) | ✗ Not LOC | Adjective, not a location name |
| `Батыс` ("the West" as geopolitical concept) | LOC if clearly referring to a defined region | Context-dependent |

---

## 4. Central Asian Language–Specific Guidance

### 4.1 Morphological Inflection
All five languages are agglutinative and attach suffixes directly to proper nouns. Always annotate the **full surface token**, including case and possessive suffixes:

| Base form | Inflected form | Suffix meaning | Annotate as |
|-----------|---------------|----------------|-------------|
| `Алматы` | `Алматыда` | locative (-да) | `Алматыда` [LOC] |
| `Назарбаев` | `Назарбаевқа` | dative (-қа) | `Назарбаевқа` [PER] |
| `Ташкент` | `Toshkentdan` | ablative (-dan, Uzbek Latin) | `Toshkentdan` [LOC] |
| `Аşgabat` | `Aşgabadyň` | genitive (-yň, Turkmen) | `Aşgabadyň` [LOC] |

### 4.2 Russian-Origin Names and Loanwords
Central Asian speech frequently contains Russian proper nouns (people, places, organizations) borrowed without translation. Label these using the same rules:

- Russian person names: `Путин` [PER], `Горбачёв` [PER]
- Russian organizations: `Государственная Дума` [ORG]
- Soviet-era place names that persist: `Ленинград` [LOC]

### 4.3 Arabic and Persian Names in Tajik
Tajik (tg) uses many Arabic/Persian proper names. Apply PER/ORG/LOC as usual regardless of etymological origin:

> `Имом Бухорӣ` [PER] · `Самарқанд` [LOC] · `Ҳизби Наҳзати Исломӣ` [ORG]

### 4.4 Script Variants
- **Kazakh (kk):** Cyrillic script. Some newer texts may mix Latin. Annotate regardless of script.
- **Kyrgyz (ky):** Cyrillic.
- **Tajik (tg):** Cyrillic (Persian-influenced alphabet).
- **Turkmen (tk):** Latin script (post-1993). Proper nouns are capitalized.
- **Uzbek (uz):** Latin script (post-1995). Proper nouns are capitalized.

In Cyrillic-script languages (kk/ky/tg), capitalization is **not reliable** in ASR output. Do not rely on capitalization to identify entities — use semantic context instead.

### 4.5 Code-Switching and Mixed Language
Speakers frequently switch to Russian mid-sentence. Annotate entities in whichever language they appear:

> `ол [Россия [LOC]] президентімен [Путинмен [PER]] кездесті`  
> *(He met with the Russian president, with Putin)*

---

## 5. Speech Transcript–Specific Guidance

### 5.1 ASR Recognition Errors
Transcripts may contain **automatic speech recognition (ASR) errors**. Common issues:
- Proper nouns split or merged: `Алма Ты` instead of `Алматы`
- Name confusion: acoustically similar names substituted
- Punctuation absent or inconsistent

**Guideline:** Annotate the text **as written**, even if you suspect an ASR error. Do not mentally correct the transcript before annotating. If a name is split across two tokens due to ASR, annotate both tokens as one span.

### 5.2 Partial or Unclear Names
If an entity mention is partially garbled but the context (including audio) makes clear what entity is intended, annotate the written form.  
If neither text nor audio provides sufficient evidence, **do not annotate**.

### 5.3 Use the Audio When Uncertain
Listen to the audio clip when:
- The text is ambiguous (a word could be a common noun or proper name)
- You suspect an ASR error affecting entity boundaries
- The entity type is unclear from text alone

### 5.4 Numbers and Titles Adjacent to Names
Do not include numerals (ordinals, ranks) or titles as part of the entity span:

| Text | ✓ | ✗ |
|------|---|---|
| `Президент Назарбаев` | `Назарбаев` [PER] | `Президент Назарбаев` [PER] |
| `II Елизавета` | `Елизавета` [PER] | `II Елизавета` [PER] |
| `№1 мектеп` | ✗ (school number, no proper name) | — |

Exception: If a numeral is **part of the official proper name** (e.g., `Олимпиада-80`, `COVID-19`), include it in the span.

---

## 6. Label Selection Quick Reference

| Scenario | Label |
|----------|-------|
| Individual person's name (real or fictional) | **PER** |
| Person's name with case suffix attached | **PER** (include suffix) |
| Government body, ministry, parliament | **ORG** |
| Company, corporation, state enterprise | **ORG** |
| International organization | **ORG** |
| University, research institute | **ORG** (when acting as institution); **LOC** (when referring to the physical location) |
| Country name | **LOC** |
| City, town, village name | **LOC** |
| River, mountain, lake, sea | **LOC** |
| Street, neighborhood, district | **LOC** |
| Ethnic group acting as collective agent | **ORG** |
| Nationality adjective | ✗ Not labeled |
| Generic noun (city, company, river) without name | ✗ Not labeled |
| Title or honorific alone | ✗ Not labeled |

---

## 7. Difficult Cases and FAQ

**Q: Should I annotate a country name that is also the name of the government?**  
A: Use context. If the sentence refers to the nation/territory (`Қазақстанда өткен` = "held in Kazakhstan"), label LOC. If it refers to the government as an actor (`Қазақстан шешім қабылдады` = "Kazakhstan decided"), label LOC (states-as-actors are conventionally LOC in most NER standards, not ORG, unless the specific government body name is given).

**Q: A person's name also appears as a city name (e.g., a city named after a person). Which label?**  
A: Use the referent in context. If referring to the city → LOC. If referring to the person → PER.

**Q: The transcript contains only a surname. Should I label it?**  
A: Yes, if context makes clear it refers to a specific individual (e.g., in a news context where the person was introduced earlier). Otherwise, use caution.

**Q: What about sports teams?**  
A: Label as ORG (e.g., `Барселона` as a football club). If `Барселона` clearly refers to the city → LOC.

**Q: Organization names containing location names?**  
A: Label separately. `Алматы қалалық мәслихаты` → `Алматы` [LOC] + `Алматы қалалық мәслихаты` [ORG]. In practice, annotate the full ORG span; the embedded LOC need not be separately labeled unless it appears independently elsewhere.

**Q: Should I annotate URLs, email addresses, or hashtags?**  
A: No. These are metadata, not named entities.

**Q: A speaker mentions "our president" or "the minister" without naming them. Do I label?**  
A: No. Generic role references without a proper name are not labeled.

---

## 8. Annotation Quality Checklist

Before submitting each annotation session, verify:

- [ ] All person names are labeled PER (including inflected forms)
- [ ] Organization names are complete (not just the keyword; full official name span)
- [ ] Location names include all tokens of multi-word toponyms
- [ ] No titles, generic nouns, or adjectives are included in spans
- [ ] Morphological suffixes are included as part of the entity span
- [ ] Audio was consulted for any uncertain entity
- [ ] Spurious LLM pre-annotations have been removed
- [ ] Missed entities have been added

---

## 9. Inter-Annotator Agreement

To ensure consistency across annotators:
- Target **Cohen's κ ≥ 0.80** at the entity-level for final test set annotations
- Disagreements are resolved by a senior annotator
- Edge cases not covered by these guidelines should be flagged and added as FAQ entries for future versions

---

## Appendix A: Label Summary

| Label | Full Name | Cyrillic Examples | Latin Examples |
|-------|-----------|------------------|----------------|
| PER | Person | `Назарбаев`, `Айтматов`, `Раҳмон` | `Berdimuhamedow`, `Navoiy` |
| ORG | Organization | `БҰҰ`, `Жогорку Кеңеш`, `Ҳизби Наҳзат` | `BMT`, `NATO` |
| LOC | Location | `Қазақстанда`, `Бишкекке`, `Душанбе` | `Toshkentda`, `Aşgabatda` |

## Appendix B: Common Abbreviations by Language

| Abbreviation | Full name | Language | Type |
|-------------|-----------|----------|------|
| БҰҰ | Біріккен Ұлттар Ұйымы (UN) | kk | ORG |
| ШЫҰ | Шанхай ынтымақтастық ұйымы (SCO) | kk | ORG |
| ДДҰ | Дүниежүзілік денсаулық сақтау ұйымы (WHO) | kk | ORG |
| БУУ | Бириккен Улуттар Уюму (UN) | ky | ORG |
| СММ | Созмони Миллали Муттаҳид (UN) | tg | ORG |
| BMG | Birleşen Milletler Guramasy (UN) | tk | ORG |
| BMT | Birlashgan Millatlar Tashkiloti (UN) | uz | ORG |
| ТМД | Тәуелсіз Мемлекеттер Достастығы (CIS) | kk | ORG |
| МВФ / IMF | International Monetary Fund | all | ORG |

---

*CA-STNER Annotation Guidelines v1.0*  
*For questions or guideline updates, contact the dataset maintainers.*
