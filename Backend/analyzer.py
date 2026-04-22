"""
ResumeIQ — analyzer.py  (v4 · ATS Intelligence Engine)
═══════════════════════════════════════════════════════════════════════════════

Domain-agnostic resume scoring engine. The JD is the ONLY reference point.

THE CENTRAL FIX IN THIS VERSION
────────────────────────────────
Previous builds extracted random frequent words (collaborating, efficient,
managing) as "skills". This version implements a strict SKILL GATE:

  A token is a skill only if it:
    ✔ is a known tool / technology / platform (matched against a broad catalog)
    ✔ is a domain-specific concept (detected via structural heuristics)
    ✔ appears as a meaningful multi-word phrase from the JD (bigram/trigram)
    ✔ passes ALL of the following reject filters:
        ✘ does NOT end with -ing / -ed / -ly / -tion / -ment (verb/adj forms)
        ✘ is NOT in the generic non-skill word list
        ✘ is NOT a pure common English noun with no technical meaning
        ✘ is NOT shorter than 2 characters

Pipeline
────────
  §0  Constants: stop-words, non-skill blocklist, skill catalog
  §1  Text utilities: clean / tokenise / n-gram
  §2  Domain detection  (JD-only, cluster scoring)
  §3  Skill gate        (the core fix — strict token classifier)
  §4  JD skill extraction  (freq + position + phrase)
  §5  Tool extraction   (CamelCase / acronym heuristics)
  §6  Synonym expansion  (canonical concept matching)
  §7  TF-IDF cosine similarity  (stdlib only)
  §8  Six scoring factors
  §9  Keyword-stuffing penalty
  §10 Domain-aware weight adjustment
  §11 Intelligent suggestions
  §12 Main entry point: analyze_resume()

Author  : ResumeIQ  |  Python 3.9+  |  stdlib only (re, math, collections)
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import NamedTuple


# ═══════════════════════════════════════════════════════════════════════════
#  §0  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Stop-words: removed before ANY processing ──────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    # Function words
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','into','through','about','up','down','out','off','over',
    'under','again','then','once','here','there','when','where','why',
    'how','what','which','who','whom','than','too','very','just','also',
    'well','between','such','while','during','before','after','above',
    'below','across','within','without','around','near','among','per',
    'via','vs','ie','eg',
    # Auxiliaries
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'shall','can','must','need','am',
    # Pronouns
    'i','me','my','we','our','you','your','they','their','it','its',
    'he','she','him','her','us','this','that','these','those',
    # Job-posting boilerplate (these are NEVER skills)
    'candidate','position','role','job','company','team','business',
    'required','preferred','responsibilities','qualifications','opportunity',
    'looking','seeking','join','help','excellent','strong','great','highly',
    'minimum','maximum','able','good','new','include','includes','including',
    'following','related','various','other','basis','etc','plus',
    'day','time','daily','weekly','monthly','annual',
    'year','years','month','months','experience','level','high','low',
    'right','far','even','still','back','give','make','take','see',
    'know','get','go','come','think','look','want','need','own',
    # Generic HR terms
    'provide','support','ensure','maintain','manage','coordinate',
    'report','lead','work','build','develop','deliver','drive',
    'create','design','implement','deploy','operate','handle',
    'review','improve','achieve','measure','track','monitor',
    'communicate','present','document','maintain',
})

# ── Non-skill blocklist: words that pass stop-word filter but are NOT skills
# These are verbs, adjectives, and meaningless nouns that frequently pollute
# skill extraction systems. Comprehensive enough to cover all domains.
_NON_SKILL: frozenset[str] = frozenset({
    # -ing gerunds (actions, not skills)
    'using','working','managing','developing','building','creating',
    'designing','implementing','collaborating','communicating','leading',
    'supporting','ensuring','maintaining','providing','delivering',
    'driving','improving','achieving','tracking','monitoring','writing',
    'reading','learning','growing','scaling','optimizing','optimising',
    'reviewing','processing','handling','coordinating','presenting',
    'documenting','testing','deploying','operating','running','enabling',
    'solving','planning','executing','defining','identifying','assessing',
    'evaluating','reporting','validating','integrating','automating',
    'configuring','installing','migrating','analysing','analyzing',
    'researching','investigating','recruiting','onboarding','training',
    'mentoring','coaching','facilitating','negotiating','budgeting',
    'forecasting','auditing','reconciling','tendering','procuring',
    'supervising','inspecting','commissioning','drafting','certifying',
    # -ed past tense verbs disguised as adjectives
    'experienced','skilled','proficient','knowledgeable','motivated',
    'dedicated','focused','organised','organized','detailed','required',
    'preferred','listed','mentioned','described','provided','used',
    'established','proven','demonstrated','advanced','improved','reduced',
    'increased','delivered','designed','built','created','developed',
    # Pure adjectives (no technical meaning)
    'efficient','effective','innovative','dynamic','proactive','strong',
    'excellent','good','great','high','low','fast','quick','flexible',
    'scalable','reliable','robust','complex','simple','clear','clean',
    'modern','current','latest','recent','new','old','large','small',
    'multiple','various','different','similar','specific','general',
    'critical','important','key','core','main','primary','secondary',
    'cross','functional','technical','digital','global','local',
    # Generic non-technical nouns
    'skills','ability','abilities','knowledge','understanding','expertise',
    'experience','background','exposure','familiarity','awareness',
    'candidate','developer','developers','engineer','engineers',
    'professional','professionals','expert','experts','specialist',
    'individual','person','people','member','members','staff',
    'application','applications','system','systems','solution','solutions',
    'platform','process','processes','service','services','product','products',
    'environment','environments','tool','tools','technology','technologies',
    'framework','frameworks','library','libraries','language','languages',
    'performance','quality','security','reliability','efficiency','accuracy',
    'standard','standards','practice','practices','procedure','procedures',
    'documentation','requirement','requirements','specification','specifications',
    'feature','features','function','functions','component','components',
    'module','modules','interface','interfaces','model','models',
    'data','information','report','reports','analysis','result','results',
    'output','input','issue','issues','problem','problems','challenge',
    'task','tasks','project','projects','initiative','initiatives',
    'goal','goals','objective','objectives','target','targets',
    'impact','value','benefit','benefits','opportunity','opportunities',
    'area','areas','field','domain','sector','industry','market',
    'client','clients','customer','customers','user','users','stakeholder',
    'stakeholders','partner','partners','vendor','vendors',
    # Numbers / units that can appear
    'years','months','plus','least','least','level','levels',
    # Domain/role descriptor words (NOT skills)
    'civil','mechanical','software','digital','senior','junior','mid','lead',
    'entry','principal','staff','associate','assistant','intern','head',
    'site','field','office','remote','hybrid','contract','permanent',
    'full','part','time','based','driven','focused','oriented',
    'code','codes','event','events','streaming','pipeline','pipelines',
})

# ── Suffix patterns that almost always indicate non-skill words ─────────────
_NON_SKILL_SUFFIXES: re.Pattern = re.compile(
    r'^.{3,}(ing|ings|tion|tions|sion|sions|ment|ments|ness|nesses'
    r'|ity|ities|ance|ances|ence|ences|ive|ives|ous|ful|less|ward|wise'
    r'|ably|ibly|ally|edly|ingly|ionally)$'
)

# EXCEPTION: some -ing / -tion words ARE valid technical concepts
# (e.g., "networking", "programming", "automation", "containerization")
_SUFFIX_EXCEPTIONS: frozenset[str] = frozenset({
    'programming','networking','containerization','containerisation',
    'virtualization','virtualisation','automation','configuration',
    'orchestration','authentication','authorization','authorisation',
    'encryption','compression','pagination','serialization','serialisation',
    'microservices','monitoring','testing',
    # domain concepts that look like gerunds
    'accounting','engineering','manufacturing','marketing','processing',
    'computing','consulting',
})

# ── Broad skill catalog: anchors the skill gate.
#    If a JD token appears in ANY of these sets, it is definitely a skill.
#    Organised by domain but intentionally cross-domain to avoid bias.
#    This is used for CLASSIFICATION only — not for hardcoded skill lists.
_SKILL_CATALOG: frozenset[str] = frozenset({
    # ── Programming languages ──
    'python','javascript','typescript','java','golang','rust','ruby','php',
    'swift','kotlin','scala','perl','bash','shell','powershell',
    'html','css','sql','graphql','solidity','dart','matlab','cobol',
    'fortran','haskell','elixir','clojure','groovy','julia','lua',
    'objectivec','asm','vba','apex','abap',
    # ── Web / backend frameworks ──
    'react','angular','vuejs','vue','svelte','nextjs','nuxtjs','gatsby',
    'django','flask','fastapi','rails','laravel','spring','express',
    'nestjs','actix','gin','fiber','asp.net','symfony','codeigniter',
    'webpack','vite','babel','tailwind','bootstrap','sass','scss',
    # ── Mobile ──
    'android','ios','flutter','reactnative','xamarin','ionic','cordova',
    # ── Databases ──
    'postgresql','mysql','sqlite','oracle','mssql','mariadb',
    'mongodb','redis','cassandra','dynamodb','couchdb','firestore',
    'elasticsearch','solr','neo4j','influxdb','clickhouse','snowflake',
    'bigquery','redshift','databricks','hive','hbase','supabase',
    'prisma','sequelize','sqlalchemy','hibernate',
    # ── Cloud platforms ──
    'aws','azure','gcp','digitalocean','heroku','vercel','netlify',
    'cloudflare','linode','vultr','ibmcloud','oraclecloud',
    # ── Cloud services / infra ──
    'ec2','s3','lambda','ecs','eks','rds','sqs','sns','cloudwatch',
    'iam','vpc','route53','cloudfront','cognito','api gateway',
    'blob storage','app service','azure functions','cloud run',
    # ── DevOps / CI-CD ──
    'ci/cd','cicd','devops','mlops','dataops','gitops','devsecops',
    'docker','kubernetes','helm','terraform','ansible','vagrant','packer',
    'jenkins','github actions','gitlab ci','circleci','travis ci',
    'azure pipelines','teamcity','bitbucket pipelines','argocd','flux',
    'puppet','chef','saltstack','prometheus','grafana','datadog',
    'newrelic','splunk','elk','logstash','kibana','jaeger',
    # ── Version control ──
    'git','github','gitlab','bitbucket','svn','mercurial',
    # ── Data engineering / ML / AI ──
    'spark','kafka','airflow','dbt','flink','beam','nifi','luigi',
    'mlflow','kubeflow','sagemaker','vertex ai','azureml',
    'tensorflow','pytorch','keras','scikit-learn','xgboost','lightgbm',
    'catboost','huggingface','transformers','langchain','openai',
    'pandas','numpy','scipy','matplotlib','seaborn','plotly','bokeh',
    'tableau','powerbi','looker','qlik','metabase','superset',
    # ── Testing ──
    'selenium','cypress','playwright','puppeteer','jest','mocha','chai',
    'pytest','junit','testng','nunit','phpunit','cucumber','postman',
    'jmeter','locust','gatling','appium',
    # ── Project / workflow tools ──
    'jira','confluence','trello','asana','notion','monday','basecamp',
    'smartsheet','airtable','clickup','linear','github projects',
    # ── Networking ──
    'tcp/ip','dns','http','https','ssl','tls','vpn','tcp','udp',
    'nginx','apache','haproxy','istio','envoy','consul','vault',
    # ── Security ──
    'owasp','oauth','jwt','saml','ldap','sso','keycloak','auth0',
    'penetration testing','pen testing','soc','siem','iam',
    # ── CAD / engineering tools ──
    'autocad','solidworks','catia','revit','civil3d','inventor',
    'microstation','archicad','bentley','staad','etabs','safe',
    'sap2000','robot structural','primavera','p6','ms project',
    'microsoft project','pert','cpm','navisworks','tekla',
    # ── Mechanical / manufacturing ──
    'ansys','abaqus','nastran','hypermesh','fea','fem','cfd',
    'cnc','cam','gd&t','plc','scada','hvac','piping',
    'autocad plant','pdms','sp3d','aveva',
    # ── Finance / accounting ──
    'sap','oracle','tally','quickbooks','xero','sage','netsuite',
    'dynamics365','hyperion','ifrs','gaap','cfa','acca','cpa',
    'bloomberg','reuters','morningstar','excel','vba','power query',
    'pivot tables','financial modeling','dcf','lbo','var',
    # ── Healthcare ──
    'epic','cerner','meditech','athenahealth','allscripts','ehr',
    'emr','hl7','fhir','hipaa','icd10','cpt','dicom',
    # ── GIS / surveying ──
    'arcgis','qgis','mapinfo','leica','trimble','gps','gnss',
    # ── Domain-specific technical concepts ──
    'rcc','rcc design','structural analysis','quantity surveying',
    'boq','bill of quantities','rate analysis','contract management',
    'procurement','geotechnical','soil testing','mix design',
    'machine learning','deep learning','neural network','nlp',
    'computer vision','reinforcement learning','statistical analysis',
    'regression','classification','clustering','feature engineering',
    'api','rest api','graphql api','grpc','soap','websocket',
    'microservices','monolith','serverless','event driven','cqrs','ddd',
    'agile','scrum','kanban','lean','six sigma','pmp','prince2',
    'blockchain','smart contract','defi','web3','nft',
    'seo','sem','google analytics','google ads','facebook ads',
    'crm','salesforce','hubspot','zendesk','marketo','mailchimp',
    'erp','mrp','wms','tms','scm',
    # ── Methodologies / certifications as skills ──
    'tdd','bdd','ddd','solid','dry','rest','mvc','mvvm',
    'iso9001','iso27001','pci dss','gdpr','sox','cmmi',
    'itil','cobit','togaf','zachman',
    # ── Soft technical skills that ARE measurable ──
    'code review','pair programming','system design','technical writing',
    'data analysis','data visualization','data modelling','erd',
})


# ═══════════════════════════════════════════════════════════════════════════
#  §1  TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """
    Normalise text:
    - lowercase
    - keep alphanumeric, hyphens (mid-word), slashes (ci/cd), +, #
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r'[^\w\s\-\/\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _base_tokens(text: str) -> list[str]:
    """
    Tokenise cleaned text: strip leading/trailing punctuation per token,
    remove stop-words, require length ≥ 2, require at least one letter.
    Does NOT apply the skill gate — raw tokens for general processing.
    """
    tokens: list[str] = []
    for w in _clean(text).split():
        w = w.strip('-/#+ ')
        if (
            w
            and w not in _STOPWORDS
            and len(w) >= 2
            and re.search(r'[a-z]', w)
            and not w.isdigit()
        ):
            tokens.append(w)
    return tokens


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """Build n-grams from a token list."""
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _word_count(text: str) -> int:
    return len(text.split())


def _unique_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  §2  DOMAIN DETECTION  (JD-only)
# ═══════════════════════════════════════════════════════════════════════════

_DOMAIN_SIGNALS: dict[str, list[str]] = {
    'software':   ['api','backend','frontend','microservice','database','cloud',
                   'deploy','repository','git','agile','sprint','devops','sdk',
                   'endpoint','query','server side','full stack','codebase'],
    'civil':      ['rcc','concrete','structural','autocad','primavera','surveying',
                   'quantity','drawing','contract','tender','geotechnical',
                   'foundation','reinforcement','earthwork','drainage','boq'],
    'mechanical': ['solidworks','cad','machining','hydraulic','pneumatic','hvac',
                   'thermal','ansys','manufacturing','cnc','tolerance','fea',
                   'piping','fatigue','stress analysis'],
    'finance':    ['audit','ledger','reconciliation','ifrs','gaap','compliance',
                   'financial reporting','budget','forecast','equity','credit',
                   'valuation','portfolio','investment','liquidity','taxation'],
    'healthcare': ['patient','clinical','diagnosis','ehr','hipaa','pharmaceutical',
                   'dosage','therapy','surgical','nursing','icd','cpt','fhir'],
    'marketing':  ['seo','campaign','brand','content','conversion','funnel',
                   'ctr','roi','copywriting','google analytics','social media',
                   'acquisition','retention','engagement'],
    'data':       ['etl','data pipeline','warehouse','spark','airflow','databricks',
                   'snowflake','tableau','powerbi','data modelling','feature',
                   'analytics','dashboard','visualization'],
    'hr':         ['recruitment','onboarding','payroll','talent','workforce',
                   'compensation','hris','performance management','engagement',
                   'diversity','succession','labour law'],
}


def detect_domain(jd_text: str) -> str:
    """Score JD against domain signal clusters; return top domain or 'general'."""
    lower  = jd_text.lower()
    scores = {d: sum(1 for s in sigs if s in lower)
              for d, sigs in _DOMAIN_SIGNALS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'general'


# ═══════════════════════════════════════════════════════════════════════════
#  §3  SKILL GATE  — the core fix
#
#  A token passes the gate if it satisfies AT LEAST ONE of:
#    (A) exact membership in _SKILL_CATALOG
#    (B) matches a CamelCase / ALL-CAPS / versioned-name pattern
#        (heuristic for unknown tools like "AutoCAD", "SAP", "PowerBI")
#    (C) is a multi-word phrase (spaces) — phrases bypass unigram filters
#        because they carry enough context to be self-evidently meaningful
#
#  AND fails NONE of:
#    (X1) in _NON_SKILL blocklist
#    (X2) matched by _NON_SKILL_SUFFIXES (unless in _SUFFIX_EXCEPTIONS)
#    (X3) length < 2
# ═══════════════════════════════════════════════════════════════════════════

# Pattern: looks like a proper-noun tool name in original (pre-lowercased) text
_TOOL_PATTERN = re.compile(
    r'^('
    r'[A-Z][a-z]+(?:[A-Z][a-z]+)+'    # CamelCase: TensorFlow, FastAPI
    r'|[A-Z]{2,8}'                      # Acronyms: AWS, SQL, JIRA
    r'|[A-Za-z]+\d+\.?\d*'             # Versioned: AutoCAD2024, Python3
    r'|[A-Za-z]+\s?[Pp]\d+'            # Primavera P6, P3
    r')$'
)


def _is_skill_token(token: str, original_jd: str = '') -> bool:
    """
    Return True if `token` is a genuine skill/tool/technology.
    `token` should already be lowercased and stripped.
    `original_jd` (raw case) is used for the tool-pattern heuristic.

    Check order (deliberate):
      1. Multi-word phrases → immediate True (bypass unigram filters)
      2. Hard rejects: blocklist, length, suffix pattern
      3. Hard accepts: skill catalog, CamelCase tool, short acronym
    """
    # ── Step 1: Multi-word phrases bypass all unigram filters ───────────
    # A phrase like "machine learning" / "quantity surveying" / "rest api"
    # is inherently contextual and meaningful — pass immediately.
    if ' ' in token:
        # But still reject if ALL words are non-skills / stopwords
        parts = token.split()
        non_junk = [
            p for p in parts
            if p not in _STOPWORDS
            and p not in _NON_SKILL
            and len(p) >= 2
            and not re.fullmatch(r'[a-z]{3,}(ing|ed|ly)$', p)
        ]
        return len(non_junk) >= 1   # at least one meaningful word

    # ── Step 2: Hard rejects (unigrams only below this point) ───────────

    # Reject: length < 2
    if len(token) < 2:
        return False

    # Reject: in non-skill blocklist
    if token in _NON_SKILL:
        return False

    # Reject: suffix pattern indicating verb/adjective (with exceptions)
    if (
        _NON_SKILL_SUFFIXES.match(token)
        and token not in _SUFFIX_EXCEPTIONS
    ):
        return False

    # ── Step 3: Hard accepts ─────────────────────────────────────────────

    # Accept: in skill catalog (definitive)
    if token in _SKILL_CATALOG:
        return True

    # Accept: looks like a proper-noun tool in original text (CamelCase / ACRONYM)
    if original_jd:
        for m in re.finditer(re.escape(token), original_jd, re.IGNORECASE):
            if _TOOL_PATTERN.match(original_jd[m.start():m.end()]):
                return True

    # Accept: short ALL-CAPS acronym (2-5 letters) not in reject list
    if re.fullmatch(r'[a-z]{2,5}', token) and token.upper() not in _ACRONYM_REJECTS:
        # Extra guard: must look like a real acronym (not a common word)
        common_short_words = {
            'the','and','for','are','but','not','you','all','can',
            'her','was','one','our','out','day','get','has','him',
            'his','how','its','may','now','old','see','two','way',
            'who','own','man','new','top','big','key','few','yet',
            'ago','off','run','set','far','use','put','try','add',
            'ask','buy','cut','end','hit','let','say','sit','win',
        }
        if token not in common_short_words:
            return True

    return False


def _filter_to_skills(tokens: list[str], original_jd: str = '') -> list[str]:
    """Apply the skill gate to a list of tokens."""
    return [t for t in tokens if _is_skill_token(t, original_jd)]


# ═══════════════════════════════════════════════════════════════════════════
#  §4  JD SKILL EXTRACTION  (frequency + position + phrase detection)
# ═══════════════════════════════════════════════════════════════════════════

def extract_jd_skills(jd_text: str, top_n: int = 25) -> list[str]:
    """
    Extract REAL skills, tools, and domain concepts from the JD.

    Critical design:
    - N-grams are built ONLY from skill-passing unigrams (no generic words).
      This prevents phrases like "backend engineer python expertise" which are
      sentence fragments, not skills.
    - Phrases use PROXIMITY: two skill tokens that are adjacent (or separated
      by at most one stop-word) in the original JD form a candidate bigram.
    - Position weighting: skills in the first third of JD score higher.
    """
    jd_lower = jd_text.lower()
    words    = jd_text.split()
    third    = max(len(words) // 3, 1)

    def _pos_weight(phrase: str) -> float:
        idx = jd_lower.find(phrase.lower())
        if idx < 0:
            return 0.8
        return 1.4 if len(jd_text[:idx].split()) < third else 1.0

    # ── Step A: Unigram skill extraction ──────────────────────────────────
    all_toks  = _base_tokens(jd_text)
    tok_freq  = Counter(all_toks)
    uni_skills = [w for w in tok_freq if _is_skill_token(w, jd_text)]

    skill_set = set(uni_skills)

    # ── Step B: Proximity-based bigrams (skill + skill, adjacent in JD) ───
    # Tokenise JD preserving order, mark each token as skill/non-skill
    clean_words = [w.strip("-/#+ ") for w in jd_lower.split()]
    bigrams: list[str] = []
    trigrams: list[str] = []

    for i, w in enumerate(clean_words):
        if w not in skill_set:
            continue
        # Look ahead for another skill within a window of 2 tokens
        for j in range(i + 1, min(i + 3, len(clean_words))):
            w2 = clean_words[j]
            if w2 in skill_set:
                phrase = f"{w} {w2}"
                # Only form bigram if both tokens are real skills individually
                if _is_skill_token(w, jd_text) and _is_skill_token(w2, jd_text):
                    bigrams.append(phrase)
                break  # only grab the nearest next skill

        # Trigrams: skill + skill + skill within window of 3
        for j in range(i + 1, min(i + 4, len(clean_words))):
            if clean_words[j] in skill_set:
                for k in range(j + 1, min(j + 3, len(clean_words))):
                    if clean_words[k] in skill_set:
                        phrase = f"{w} {clean_words[j]} {clean_words[k]}"
                        trigrams.append(phrase)
                        break
                break

    # ── Step C: Also detect known multi-word skills directly in JD ────────
    catalog_phrases: list[str] = []
    for skill in _SKILL_CATALOG:
        if ' ' in skill and skill in jd_lower:
            catalog_phrases.append(skill)

    # For synonym map — if canonical multi-word concept is in JD
    for canonical, synonyms in _SYNONYM_MAP.items():
        if ' ' in canonical and canonical in jd_lower:
            catalog_phrases.append(canonical)
        for syn in synonyms:
            if ' ' in syn and syn in jd_lower:
                catalog_phrases.append(syn)

    # ── Step D: Score and rank all candidates ─────────────────────────────
    scored: list[tuple[str, float]] = []

    # Trigrams (highest weight)
    tri_freq = Counter(trigrams)
    for phrase, cnt in tri_freq.items():
        scored.append((phrase, cnt * _pos_weight(phrase) * 2.5))

    # Known catalog phrases (authoritative)
    for phrase in set(catalog_phrases):
        freq_in_jd = jd_lower.count(phrase)
        scored.append((phrase, (freq_in_jd + 1) * _pos_weight(phrase) * 2.2))

    # Bigrams
    bi_freq = Counter(bigrams)
    for phrase, cnt in bi_freq.items():
        if not any(phrase in t for t in tri_freq):
            scored.append((phrase, cnt * _pos_weight(phrase) * 1.8))

    # Unigrams
    for w in uni_skills:
        # Skip if already covered by a phrase
        if not any(w in p.split() for p in
                   list(tri_freq.keys()) + list(bi_freq.keys()) + catalog_phrases):
            scored.append((w, tok_freq[w] * _pos_weight(w) * 1.0))

    # ── Step E: Sort, deduplicate, return ─────────────────────────────────
    scored.sort(key=lambda x: -x[1])

    seen_words: set[str] = set()
    result: list[str] = []

    for phrase, _ in scored:
        parts = set(phrase.split())
        # For multi-word: always include (provide context)
        # For unigrams: skip if already subsumed by a phrase in result
        if len(parts) == 1 and parts.issubset(seen_words):
            continue
        seen_words.update(parts)
        result.append(phrase)
        if len(result) >= top_n:
            break

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §5  TOOL EXTRACTION  (CamelCase / acronym heuristic, JD-driven)
# ═══════════════════════════════════════════════════════════════════════════

# Common English words that look like acronyms but are NOT tools
_ACRONYM_REJECTS: frozenset[str] = frozenset({
    'THE','AND','FOR','WITH','FROM','THAT','THIS','WILL','HAVE',
    'BEEN','THEY','THEIR','WHEN','WHERE','WHAT','WHICH','THAN',
    'INTO','OVER','ALSO','SUCH','ABLE','WELL','VERY','BOTH','SAME',
    'EACH','MORE','MOST','SOME','MANY','MUCH','LESS','LONG','FULL',
    'BEST','MAIN','HIGH','FAST','GOOD','NEED','MUST','MAKE','TAKE',
    'YOU','WAS','ARE','HAD','HAS','OUR','CAN','MAY','NOT',
})

_TOOL_RE: list[re.Pattern] = [
    re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'),          # CamelCase
    re.compile(r'\b[A-Z]{2,7}\b'),                             # ACRONYM
    re.compile(r'\b[A-Za-z]+\s[Pp]\d+\b'),                    # Primavera P6
    re.compile(r'\b[A-Za-z]{3,}\s?\d+(?:\.\d+)?\b'),          # AutoCAD 2024
    re.compile(r'\b[A-Za-z]+(?:\.[a-zA-Z]+)+\b'),             # asp.net, node.js
    re.compile(r'\b[A-Za-z]+(?:\/[A-Za-z]+)+\b'),             # CI/CD, TCP/IP
]


def extract_tools_from_jd(jd_text: str) -> list[str]:
    """
    Detect tool-like tokens using structural cues (capitalisation, versioning).
    Works for any domain (AutoCAD, TensorFlow, SAP, JIRA, Primavera P6...).
    Returns lowercased, deduplicated list.
    """
    found: set[str] = set()
    for pat in _TOOL_RE:
        for m in pat.finditer(jd_text):
            tok = m.group()
            upper = tok.upper().strip()
            # Reject common English words masquerading as acronyms
            if upper in _ACRONYM_REJECTS:
                continue
            lower_tok = tok.lower().strip()
            # Must pass the skill gate too
            if _is_skill_token(lower_tok, jd_text):
                found.add(lower_tok)

    # Also add anything from the skill catalog present in JD
    jd_lower = jd_text.lower()
    for skill in _SKILL_CATALOG:
        if skill in jd_lower:
            found.add(skill)

    return sorted(found)


# ═══════════════════════════════════════════════════════════════════════════
#  §6  SYNONYM & CONCEPT EXPANSION
# ═══════════════════════════════════════════════════════════════════════════

# canonical concept → list of equivalent expressions (any domain)
_SYNONYM_MAP: dict[str, list[str]] = {
    # DevOps / CI-CD
    'ci/cd':                ['github actions','gitlab ci','jenkins','circleci',
                             'travis ci','azure pipelines','teamcity',
                             'bitbucket pipelines','argocd'],
    'version control':      ['git','github','gitlab','bitbucket','svn','mercurial'],
    # Cloud
    'cloud platform':       ['aws','azure','gcp','google cloud','digitalocean',
                             'heroku','vercel','cloudflare'],
    'containerisation':     ['docker','kubernetes','helm','eks','aks','gke',
                             'podman','containerd'],
    # Data / BI
    'business intelligence':['tableau','powerbi','power bi','looker','qlik',
                             'metabase','superset','redash'],
    # Scheduling
    'project scheduling':   ['primavera','p6','ms project','microsoft project',
                             'gantt','oracle primavera','smartsheet'],
    # Message queues
    'message queue':        ['kafka','rabbitmq','sqs','pubsub','activemq',
                             'celery','nats'],
    # Observability
    'observability':        ['datadog','grafana','prometheus','newrelic',
                             'splunk','cloudwatch','dynatrace','elk'],
    # Databases (relational)
    'relational database':  ['postgresql','mysql','oracle','mssql','sqlite',
                             'mariadb','db2'],
    # Databases (nosql)
    'nosql database':       ['mongodb','redis','cassandra','dynamodb',
                             'couchdb','firestore','elasticsearch'],
    # CAD
    'cad software':         ['autocad','solidworks','catia','revit','civil3d',
                             'inventor','microstation','bentley'],
    # ERP
    'erp system':           ['sap','oracle erp','tally','quickbooks','xero',
                             'sage','dynamics365','netsuite','odoo'],
    # ML
    'machine learning':     ['scikit-learn','tensorflow','pytorch','keras',
                             'xgboost','lightgbm','catboost','mlflow'],
    'nlp':                  ['spacy','nltk','huggingface','transformers',
                             'bert','gpt','langchain'],
    # Testing
    'automated testing':    ['selenium','cypress','playwright','jest','pytest',
                             'junit','mocha','testng','appium'],
    # Project management
    'project management':   ['jira','trello','asana','notion','monday',
                             'basecamp','clickup','linear'],
    # Finance standards
    'financial reporting':  ['ifrs','gaap','sap fi','hyperion'],
    'risk management':      ['risk assessment','var','stress testing','basel'],
    # Healthcare systems
    'ehr system':           ['epic','cerner','meditech','athenahealth','allscripts'],
    # Quantity surveying
    'quantity surveying':   ['boq','bill of quantities','rate analysis',
                             'cost estimation','schedule of rates'],
    # Structural analysis
    'structural analysis':  ['etabs','staad','safe','sap2000','ansys','robot structural'],
    # CRM
    'crm system':           ['salesforce','hubspot','zendesk','dynamics crm',
                             'freshdesk','pipedrive'],
    # Marketing analytics
    'web analytics':        ['google analytics','adobe analytics','mixpanel',
                             'amplitude','heap'],
}


def _build_synonym_lookup() -> dict[str, str]:
    """
    Build a reverse map: synonym/alias → canonical concept name.
    Used for O(1) lookup during matching.
    """
    lookup: dict[str, str] = {}
    for canonical, synonyms in _SYNONYM_MAP.items():
        lookup[canonical] = canonical
        for s in synonyms:
            lookup[s] = canonical
    return lookup


_SYN_LOOKUP: dict[str, str] = _build_synonym_lookup()


# ═══════════════════════════════════════════════════════════════════════════
#  §7  TF-IDF COSINE SIMILARITY  (pure stdlib)
# ═══════════════════════════════════════════════════════════════════════════

def _tfidf_vector(text: str, vocab: list[str]) -> dict[str, float]:
    """
    TF-IDF vector using simplified 2-doc IDF.
    TF  = raw_count / total_tokens
    IDF = log(2 / (1 + present_in_doc))  +  1   (add-1 smoothing)
    """
    toks  = _base_tokens(text)
    total = len(toks) or 1
    freq  = Counter(toks)
    return {
        term: (freq.get(term, 0) / total)
              * (math.log(2.0 / (1.0 + int(term in freq))) + 1.0)
        for term in vocab
    }


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    dot   = sum(v1[k] * v2.get(k, 0.0) for k in v1)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (norm1 * norm2) if (norm1 * norm2) else 0.0


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity in [0, 1]."""
    vocab = list(set(_base_tokens(text_a)) | set(_base_tokens(text_b)))
    if not vocab:
        return 0.0
    return _cosine(
        _tfidf_vector(text_a, vocab),
        _tfidf_vector(text_b, vocab),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §8  SCORING FACTORS
# ═══════════════════════════════════════════════════════════════════════════

# ── 8A  Keyword Alignment  (max 20) ───────────────────────────────────────
def _score_keyword(resume_text: str, jd_text: str) -> float:
    """
    Frequency-weighted token overlap over ALL meaningful tokens
    (not just skills — catches domain verbs, context words too).
    log(1+freq) weighting prevents one repeated word dominating.
    """
    jd_toks  = _base_tokens(jd_text)
    res_set  = set(_base_tokens(resume_text))
    jd_freq  = Counter(jd_toks)

    if not jd_freq:
        return 0.0

    matched_w = sum(math.log1p(jd_freq[t]) for t in set(jd_toks) if t in res_set)
    total_w   = sum(math.log1p(c) for c in jd_freq.values())
    ratio     = matched_w / total_w if total_w else 0.0
    return round(min(20.0, ratio * 20), 2)


# ── 8B  Skill Coverage  (max 20) ─────────────────────────────────────────

class _SkillResult(NamedTuple):
    score:   float
    matched: list[str]  # strengths (clean, display-ready)
    missing: list[str]  # gaps     (clean, display-ready)


def _match_skill(skill: str, res_lower: str) -> bool:
    """
    Check if `skill` is satisfied by the resume.
    Tries: (1) direct substring, (2) synonym/canonical match.
    """
    # 1. Direct match
    if skill in res_lower:
        return True

    # 2. Synonym match: if skill maps to a canonical, check all its synonyms
    canonical = _SYN_LOOKUP.get(skill)
    if canonical:
        all_aliases = _SYNONYM_MAP.get(canonical, []) + [canonical]
        if any(alias in res_lower for alias in all_aliases):
            return True

    # 3. Reverse: if resume satisfies a synonym of what the JD asked for
    #    e.g., JD says "cloud platform" → resume says "aws"
    for canon, synonyms in _SYNONYM_MAP.items():
        if skill == canon or skill in synonyms:
            if any(s in res_lower for s in synonyms) or canon in res_lower:
                return True

    return False


def _score_skills(resume_text: str, jd_skills: list[str]) -> _SkillResult:
    """
    Compare resume against JD-extracted skills (already gated as real skills).
    Returns score + display-ready matched/missing lists.
    """
    if not jd_skills:
        return _SkillResult(10.0, [], [])

    res_lower = resume_text.lower()
    matched:  list[str] = []
    missing:  list[str] = []

    for skill in jd_skills:
        (matched if _match_skill(skill, res_lower) else missing).append(skill)

    ratio = len(matched) / len(jd_skills)
    score = round(min(20.0, ratio * 20), 2)

    # Build lookup of known multi-word skills for phrase validation
    _known_phrases: set[str] = set()
    for s in _SKILL_CATALOG:
        if ' ' in s:
            _known_phrases.add(s)
    for canonical, synonyms in _SYNONYM_MAP.items():
        if ' ' in canonical:
            _known_phrases.add(canonical)
        for syn in synonyms:
            if ' ' in syn:
                _known_phrases.add(syn)

    def _display(lst: list[str], resume_ctx: str = "") -> list[str]:
        """
        Produce a clean, display-ready skill list.

        Rules:
        - Multi-word phrases: only show if they are a recognized skill phrase.
          Unknown proximity n-grams are decomposed into component unigrams.
        - Decomposed components: only shown if NOT already satisfied in resume
          (avoids "AWS" appearing as missing when AWS is clearly in the resume).
        - Deduplication: if "REST API" is shown, "REST" and "API" are suppressed.
        """
        seen: set[str] = set()      # lowercase keys already committed
        subsumed: set[str] = set()  # individual words covered by a phrase
        out:  list[str] = []

        multi_known: list[str] = []
        decomposed_singles: list[str] = []

        for item in sorted(lst, key=len, reverse=True):
            k = item.lower().strip()
            if not k or len(k) < 2:
                continue

            if ' ' in k:
                is_known = (
                    k in _known_phrases
                    or any(
                        k == canon or k in syns
                        for canon, syns in _SYNONYM_MAP.items()
                    )
                )
                if is_known:
                    multi_known.append(k)
                else:
                    # Decompose unknown n-gram into individual valid skills
                    for part in k.split():
                        part = part.strip("-/#+ ")
                        if (
                            _is_skill_token(part, '')
                            and part not in _STOPWORDS
                            and part not in _NON_SKILL
                        ):
                            # For "missing" list: re-verify part is truly absent
                            if resume_ctx and _match_skill(part, resume_ctx):
                                continue   # actually in resume — skip from missing
                            decomposed_singles.append(part)
            else:
                if _is_skill_token(k, '') and k not in _NON_SKILL:
                    if k not in seen:
                        seen.add(k)
                        out.append(_pretty_skill(k))

        # Commit known phrases — track their words as subsumed
        for k in multi_known:
            if k not in seen:
                seen.add(k)
                subsumed.update(k.split())
                out.append(_pretty_skill(k))

        # Commit decomposed singles only if not subsumed by a phrase
        for k in decomposed_singles:
            if k not in seen and k not in subsumed:
                seen.add(k)
                out.append(_pretty_skill(k))

        # Sort: multi-word first (more specific), then alphabetical
        out.sort(key=lambda x: (-len(x.split()), x.lower()))
        return out[:12]

    return _SkillResult(score, _display(matched, resume_text.lower()), _display(missing, resume_text.lower()))


def _pretty_skill(skill: str) -> str:
    """
    Display-friendly capitalisation for skills.
    Known acronyms → uppercase. Multi-word → title case. Single → title case.
    """
    _UPPER_SKILLS = {
        'aws','gcp','api','sql','css','html','php','sdk','ide','crm',
        'erp','ehr','emr','seo','sem','roi','ctr','cpm','kpi','bi',
        'rcc','cad','cam','fea','bim','hse','pmp','ifrs','gaap','cfa',
        'acca','cpa','var','ci/cd','tcp/ip','dns','ssl','tls','vpn',
        'rest','soap','grpc','jwt','sso','iam','mfa','gdpr','sox',
        'cpm','pert','boq','nlp','llm','sla','tdd','bdd','ddd','mvc',
        'orm','cdn','ddos','devops','mlops','dataops','gitops','finops',
    }
    if skill.lower() in _UPPER_SKILLS:
        return skill.upper()
    # Multi-word: title case each part
    return ' '.join(
        w.upper() if w.lower() in _UPPER_SKILLS else w.capitalize()
        for w in skill.split()
    )


# ── 8C  Tools & Technologies  (max 15) ────────────────────────────────────
def _score_tools(resume_text: str, jd_tools: list[str]) -> float:
    """
    Match JD-extracted tool tokens against resume.
    No artificial floor — a resume with no relevant tools scores 0.
    """
    if not jd_tools:
        return 7.5   # neutral when JD doesn't mention specific tools

    res_lower = resume_text.lower()
    matched   = sum(1 for t in jd_tools if _match_skill(t, res_lower))
    ratio     = matched / len(jd_tools)

    # Tiered scoring — not purely linear:
    # < 25% match: 0–4  | 25–60%: 4–10  | > 60%: 10–15
    if ratio >= 0.60:
        sc = 10.0 + (ratio - 0.60) / 0.40 * 5.0
    elif ratio >= 0.25:
        sc = 4.0  + (ratio - 0.25) / 0.35 * 6.0
    else:
        sc = ratio / 0.25 * 4.0

    return round(min(15.0, sc), 2)


# ── 8D  Experience Depth  (max 20) ────────────────────────────────────────

# Universal action verbs — domain-agnostic, PRESENT in resume (past tense too)
_ACTION_VERB_ROOTS: frozenset[str] = frozenset({
    'develop','build','design','implement','deliver','launch','deploy',
    'create','establish','found','initiate','introduce','architect',
    'optimise','optimize','improve','enhance','upgrade','refactor',
    'reduce','increase','boost','accelerate','streamline','simplify',
    'lead','manage','supervise','mentor','coach','direct','head',
    'coordinate','facilitate','oversee','guide','train',
    'analyse','analyze','evaluate','assess','audit','investigate',
    'research','monitor','track','measure','validate',
    'engineer','configure','automate','integrate','migrate',
    'consolidate','scale','secure','test','model','simulate',
    'fabricate','install','commission','procure','negotiate',
    'tender','budget','forecast','reconcile','approve','certify',
    'inspect','draft','spearhead','orchestrate','pioneer','drive',
})

# Match verb roots + common inflections
_ACTION_VERB_RE: re.Pattern = re.compile(
    r'\b('
    + '|'.join(re.escape(v) for v in sorted(_ACTION_VERB_ROOTS, key=len, reverse=True))
    + r')(d|ed|s|es|ing|tion|ment)?\b',
    re.IGNORECASE
)

_METRIC_RE: list[re.Pattern] = [re.compile(p, re.I) for p in [
    r'\d+\s*%',
    r'[\$£€₹]\s*[\d,]+',
    r'\b\d+\s*(million|billion|crore|lakh|thousand|[km])\b',
    r'\b\d{1,4}\s*(users|clients|customers|engineers|sites|projects|teams|contracts|vendors)\b',
    r'(reduced|increased|improved|saved|cut|grew|boosted)\s.{1,50}\d',
    r'\d+\s*(x|times)\s*(faster|cheaper|more|larger|smaller)',
    r'top\s+\d+\s*%',
    r'\d+\+?\s*(years?|yrs?)',
]]


def _score_experience(resume_text: str) -> tuple[float, bool, int]:
    """
    Returns (score /20, has_metrics, verb_count).
    Breakdown: verbs(9) + metrics(6) + depth(3) + yoe(2) = 20 max.
    """
    verb_matches = _ACTION_VERB_RE.findall(resume_text)
    unique_verbs = set(v[0].lower() for v in verb_matches)
    n_verbs      = len(unique_verbs)
    v_sc = min(9.0, n_verbs * 1.1)   # 1.1 per unique root, 9 verbs = max

    metrics_found = any(p.search(resume_text) for p in _METRIC_RE)
    m_sc = 6.0 if metrics_found else 0.0

    # Only count lines > 20 chars to skip section headers
    sentences = [s.strip() for s in re.split(r'[\n\.]', resume_text) if len(s.strip()) > 20]
    if sentences:
        avg_len = sum(len(_base_tokens(s)) for s in sentences) / len(sentences)
        r_sc    = min(3.0, avg_len / 10.0 * 3.0)  # >= 10 avg tokens = full 3pts
    else:
        r_sc = 0.0

    yoe  = len(re.findall(r'\b\d+\+?\s*(?:years?|yrs?)\b', resume_text, re.I))
    y_sc = min(2.0, yoe * 0.7)

    return round(min(20.0, v_sc + m_sc + r_sc + y_sc), 2), metrics_found, n_verbs
_SECTION_MARKERS: dict[str, list[str]] = {
    'experience': ['experience','employment','work history','career history',
                   'positions held','professional experience'],
    'skills':     ['skills','technical skills','core skills','competencies',
                   'expertise','key skills','skill set'],
    'education':  ['education','qualification','degree','academic','university',
                   'college','certifications','courses'],
    'projects':   ['projects','portfolio','personal projects','open source',
                   'case studies','notable projects'],
    'summary':    ['summary','profile','objective','about','professional summary',
                   'career summary','executive summary'],
}


def _score_quality(resume_text: str) -> tuple[float, int, dict]:
    """Returns (score /10, word_count, sections_dict)."""
    wc    = _word_count(resume_text)
    lower = resume_text.lower()

    # Length scoring (ideal 400–700 words)
    if   400 <= wc <= 700:  l_sc = 4.0
    elif 300 <= wc < 400 or 700  < wc <= 900:  l_sc = 3.0
    elif 200 <= wc < 300 or 900  < wc <= 1200: l_sc = 2.0
    else:                                       l_sc = 1.0

    # Section presence (max 6 pts, 1.2 per section)
    sections = {sec: any(kw in lower for kw in markers)
                for sec, markers in _SECTION_MARKERS.items()}
    s_sc = min(6.0, sum(sections.values()) * 1.2)

    return round(min(10.0, l_sc + s_sc), 2), wc, sections


# ── 8F  Semantic Relevance  (max 15) ──────────────────────────────────────
def _score_relevance(resume_text: str, jd_text: str) -> float:
    """TF-IDF cosine similarity, sqrt-boosted, scaled to [0, 15]."""
    cos = _cosine_similarity(resume_text, jd_text)
    return round(min(15.0, math.sqrt(max(cos, 0.0)) * 15.0), 2)


# ═══════════════════════════════════════════════════════════════════════════
#  §9  KEYWORD-STUFFING PENALTY  (0 to −10)
# ═══════════════════════════════════════════════════════════════════════════

def _stuffing_penalty(resume_text: str, jd_skills: list[str]) -> float:
    """
    Detect and penalise keyword stuffing.

    Three independent signals (max -18 total):

    Signal 1 — Low unique-word ratio        (-6 max)
      Real resumes: unique ratio typically 0.60–0.75
      Stuffed resumes: ratio < 0.45 (same words repeated)

    Signal 2 — Skills block dominance       (-5 max)
      If the Skills/Technologies section > 35% of total document,
      the resume is a keyword dump, not a narrative.

    Signal 3 — JD keyword over-density      (-7 max)
      If > 22% of all tokens are JD skills, they were pasted in —
      not written naturally. Aggressive penalisation here.

    Returns value in [-18, 0].
    """
    toks  = _base_tokens(resume_text)
    lower = resume_text.lower()
    pen   = 0.0

    # ── Signal 1: Low unique-word ratio ──────────────────────────────────
    ur = _unique_ratio(toks)
    if   ur < 0.35: pen -= 6.0
    elif ur < 0.45: pen -= 3.5
    elif ur < 0.55: pen -= 1.5

    # ── Signal 2: Oversized skills section ───────────────────────────────
    m = re.search(
        r'\b(skills?|competencies|expertise|technologies)\b.*?'
        r'(?=\b(experience|education|projects?|summary|work|employment|certif)\b|$)',
        lower, re.DOTALL | re.I,
    )
    if m:
        block_ratio = len(m.group().split()) / max(_word_count(resume_text), 1)
        if   block_ratio > 0.55: pen -= 5.0
        elif block_ratio > 0.40: pen -= 3.0
        elif block_ratio > 0.35: pen -= 1.5

    # ── Signal 3: JD keyword density ─────────────────────────────────────
    if jd_skills:
        hits    = sum(1 for s in jd_skills if s in lower)
        density = hits / max(len(toks), 1)
        if   density > 0.30: pen -= 7.0
        elif density > 0.22: pen -= 4.0
        elif density > 0.16: pen -= 1.5

    # Signal 4: skills/keyword rich BUT experience shallow = keyword dump
    # A real senior engineer has both strong skills AND real experience bullets.
    # This is computed lazily from the resume text itself.
    rough_verb_count = len(re.findall(
        r'\b(developed|built|designed|implemented|deployed|architected|optimised|'
        r'automated|created|led|managed|reduced|increased|delivered|engineered|'
        r'optimized|launched|established|streamlined|improved|configured|migrated)\b',
        lower, re.I
    ))
    tok_count = max(len(toks), 1)
    # If very few action verbs but very dense in JD keywords → keyword prose pattern
    if rough_verb_count < 3 and jd_skills:
        hits    = sum(1 for s in jd_skills if s in lower)
        density = hits / tok_count
        if density > 0.12:
            pen -= 3.0   # penalise keyword prose without any real bullets

    return max(-18.0, pen)


# ═══════════════════════════════════════════════════════════════════════════
#  §10  DOMAIN-AWARE WEIGHT ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════

_FACTOR_MAXES: dict[str, float] = {
    'keyword': 20.0, 'skills': 20.0, 'tools': 15.0,
    'experience': 20.0, 'quality': 10.0, 'relevance': 15.0,
}

# Multipliers per domain. These shift emphasis without changing total range.
_DOMAIN_MULT: dict[str, dict[str, float]] = {
    'software':   {'tools': 1.22, 'skills': 1.12, 'experience': 0.92},
    'civil':      {'experience': 1.25, 'keyword': 1.12, 'tools': 1.08},
    'mechanical': {'tools': 1.22, 'experience': 1.15},
    'finance':    {'skills': 1.20, 'experience': 1.20, 'relevance': 1.08},
    'healthcare': {'experience': 1.20, 'skills': 1.15},
    'marketing':  {'relevance': 1.22, 'keyword': 1.12},
    'data':       {'skills': 1.18, 'tools': 1.15, 'relevance': 1.10},
    'hr':         {'experience': 1.12, 'relevance': 1.10},
    'general':    {},
}


def _apply_weights(breakdown: dict[str, float], domain: str) -> dict[str, float]:
    """Apply domain multipliers; cap each factor at its defined max."""
    mults = _DOMAIN_MULT.get(domain, {})
    out: dict[str, float] = {}
    for k, v in breakdown.items():
        if k == 'penalty':
            out[k] = v
        else:
            out[k] = round(min(_FACTOR_MAXES.get(k, 20.0), v * mults.get(k, 1.0)), 2)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  §11  INTELLIGENT SUGGESTIONS  (personalised, never robotic)
# ═══════════════════════════════════════════════════════════════════════════

def _build_suggestions(
    *,
    score:         float,
    missing:       list[str],
    breakdown:     dict[str, float],
    metrics_found: bool,
    verb_count:    int,
    wc:            int,
    sections:      dict[str, bool],
    jd_text:       str,
    domain:        str,
    jd_skills:     list[str],
    jd_tools:      list[str],
    resume_lower:  str,
) -> list[str]:

    sug: list[str] = []

    # ── 1. Top missing skill — specific project suggestion ──────────
    # Clean missing list to only single-word or known phrase skills for display
    display_missing = [
        m for m in missing
        if (' ' not in m.lower())  # single word — always safe
        or m.lower() in _SYNONYM_MAP  # known canonical concept
        or any(m.lower() in synonyms for synonyms in _SYNONYM_MAP.values())
    ]

    if display_missing:
        s1 = display_missing[0]
        sug.append(
            f"The JD heavily emphasises '{s1}'. "
            f"Add a concrete bullet: e.g., 'Utilised {s1} to [outcome] in [context].' "
            "If you have indirect experience, frame it explicitly — don't leave it implicit."
        )
    elif missing:
        # Fall back to raw missing list
        s1 = missing[0]
        sug.append(
            f"Key requirement '{s1}' is missing from your resume. "
            "Add context showing you have worked with this or equivalent technology."
        )

    # ── 2. More missing skills — integration advice ─────────────────
    if len(missing) > 1:
        rest = ', '.join(f"'{x}'" for x in missing[1:4])
        sug.append(
            f"Additional gaps vs this JD: {rest}. "
            "Don't just list these in your Skills section — weave them into role bullets "
            "with context: what you used them for and what the outcome was."
        )

    # ── 3. Metrics — domain-specific examples ───────────────────────
    if not metrics_found:
        _examples = {
            'civil':      ("'Supervised RCC works worth ₹12 Cr'",
                           "'Cut material wastage by 18% via optimised planning'"),
            'mechanical': ("'Reduced component failure rate by 22% through FEA analysis'",
                           "'Optimised CNC cycle time, saving 35 min/batch'"),
            'finance':    ("'Managed ₹50 Cr budget with zero overruns'",
                           "'Reduced audit prep time by 30%'"),
            'software':   ("'Reduced API latency by 42%'",
                           "'System handled 1.2M daily active users'"),
            'data':       ("'Reduced ETL runtime from 4 h to 45 min'",
                           "'Dashboard served 200+ analysts daily'"),
            'marketing':  ("'Grew organic traffic 60% in 90 days'",
                           "'Email campaign achieved 4.8% CTR vs 2.1% industry avg'"),
            'healthcare': ("'Managed 40+ patient caseload daily with zero incident reports'",
                           "'Reduced average patient wait time by 25%'"),
            'hr':         ("'Reduced time-to-hire from 45 days to 28 days'",
                           "'Improved retention by 18% through structured onboarding'"),
        }
        ex1, ex2 = _examples.get(domain, (
            "'Achieved [result] by [N]% through [method]'",
            "'Managed [N] clients/projects/contracts simultaneously'"
        ))
        sug.append(
            f"No quantified achievements found. Add numbers to at least 3 bullets: "
            f"{ex1}, {ex2}. "
            "Hiring managers spend 6–7 seconds on initial scan — numbers make bullets stick."
        )

    # ── 4. Weak verbs ────────────────────────────────────────────────
    if verb_count < 5:
        _verb_suggestions = {
            'civil':      'Delivered, Supervised, Coordinated, Inspected, Commissioned',
            'mechanical': 'Engineered, Fabricated, Optimised, Simulated, Validated',
            'finance':    'Audited, Reconciled, Forecasted, Modelled, Evaluated',
            'software':   'Architected, Deployed, Optimised, Automated, Engineered',
            'data':       'Ingested, Transformed, Modelled, Orchestrated, Visualised',
            'marketing':  'Launched, Optimised, Grew, Converted, Analysed',
        }
        verbs = _verb_suggestions.get(domain, 'Delivered, Improved, Implemented, Analysed, Led')
        sug.append(
            f"Too few strong action verbs (found ~{verb_count}). "
            "Replace passive phrases ('was responsible for', 'helped with') "
            f"with direct openers: {verbs}. Every bullet should start with an action verb."
        )
    elif verb_count < 9:
        sug.append(
            "Some bullets lack specificity. Transform 'Worked on X improvements' → "
            "'Redesigned X, reducing [metric] by [N]'. "
            "Precision signals ownership and real expertise."
        )

    # ── 5. Missing Professional Summary ──────────────────────────────
    if not sections.get('summary'):
        top3 = [_pretty_skill(s) for s in jd_skills[:3]] if jd_skills else []
        kw_str = ', '.join(f"'{s}'" for s in top3) if top3 else 'key JD terms'
        sug.append(
            "No Professional Summary detected. "
            "Add a 3-sentence section at the very top tailored to this role. "
            f"Mirror JD language like {kw_str}. "
            "This is the first text an ATS parser and recruiter reads — "
            "it sets the matching frame for everything else."
        )

    # ── 6. Missing Projects section ──────────────────────────────────
    if not sections.get('projects') and breakdown.get('skills', 0) < 12:
        proj_skills = ', '.join(jd_skills[:3]) if jd_skills else 'key JD requirements'
        sug.append(
            "Add a Projects section to demonstrate hands-on experience. "
            f"List 2–3 projects using: {proj_skills}. "
            "Format: Project name | Tech/methods used | One measurable outcome."
        )

    # ── 7. Low semantic relevance ─────────────────────────────────────
    if breakdown.get('relevance', 0) < 7:
        phrases = [s for s in jd_skills if ' ' in s][:3]
        if phrases:
            sug.append(
                "Your resume's language doesn't closely mirror this JD. "
                f"Incorporate these exact JD phrases: {', '.join(repr(p) for p in phrases)}. "
                "ATS systems reward phrase-level matches over paraphrased equivalents."
            )
        else:
            sug.append(
                "Increase language alignment: read the JD carefully and rewrite "
                "2–3 bullets to use its exact terminology, not synonyms or paraphrases."
            )

    # ── 8. Missing tools ──────────────────────────────────────────────
    if breakdown.get('tools', 0) < 7 and jd_tools:
        gap = [_pretty_skill(t) for t in jd_tools if t not in resume_lower][:4]
        if gap:
            sug.append(
                f"Tools mentioned in JD but absent from your resume: {', '.join(gap)}. "
                "Even brief exposure (coursework, freelance, side project) is worth including — "
                "add it to your Skills section with context."
            )

    # ── 9. Word count ──────────────────────────────────────────────────
    if wc < 280:
        sug.append(
            f"Resume is only ~{wc} words — too sparse for ATS parsing. "
            "Each role should have 3–5 bullets covering: what you did, how, and the outcome. "
            "Target 400–650 words total."
        )
    elif wc > 1100:
        sug.append(
            f"At ~{wc} words your resume may be too long. "
            "Trim roles older than 7–8 years to 1–2 bullets, remove self-evident skills "
            "(e.g., 'Microsoft Word', 'email'), and target 1–2 pages maximum."
        )

    # ── 10. Stuffing warning ───────────────────────────────────────────
    if breakdown.get('penalty', 0) < -4:
        sug.append(
            "⚠ Keyword-stuffing signals detected. Low unique-word ratio or oversized "
            "Skills block found. Recruiters and ATS systems penalise this. "
            "Keep the Skills list to 8–12 relevant items and ensure each skill "
            "also appears in a contextual bullet describing HOW you used it."
        )

    return sug[:6]


# ═══════════════════════════════════════════════════════════════════════════
#  §12  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def analyze_resume(resume_text: str, jd_text: str) -> dict:
    """
    Full ATS-level domain-agnostic resume analysis.

    Parameters
    ----------
    resume_text : extracted plain text from the PDF
    jd_text     : full job description plain text

    Returns
    -------
    {
        score          : float (0–100),
        missing_skills : list[str],   # real skills only — no verbs/adjectives
        strengths      : list[str],   # real skills only
        suggestions    : list[str],   # personalised, gap-specific
        breakdown      : {
            keyword    : float,   # /20
            skills     : float,   # /20
            tools      : float,   # /15
            experience : float,   # /20
            quality    : float,   # /10
            relevance  : float,   # /15
            penalty    : float,   # 0 to -10
        }
    }
    """
    resume_lower = resume_text.lower()

    # ── §2: Domain ───────────────────────────────────────────────────
    domain = detect_domain(jd_text)

    # ── §3–§4: Skill extraction ───────────────────────────────────────
    jd_skills: list[str] = extract_jd_skills(jd_text, top_n=25)
    jd_tools:  list[str] = extract_tools_from_jd(jd_text)

    # ── §8: Score factors ─────────────────────────────────────────────
    kw_score                         = _score_keyword(resume_text, jd_text)
    sk_result                        = _score_skills(resume_text, jd_skills)
    tl_score                         = _score_tools(resume_text, jd_tools)
    ex_score, metrics_found, n_verbs = _score_experience(resume_text)
    qu_score, wc, sections           = _score_quality(resume_text)
    rel_score                        = _score_relevance(resume_text, jd_text)

    raw = {
        'keyword':    kw_score,
        'skills':     sk_result.score,
        'tools':      tl_score,
        'experience': ex_score,
        'quality':    qu_score,
        'relevance':  rel_score,
    }

    # ── §10: Domain weights ───────────────────────────────────────────
    breakdown = _apply_weights(raw, domain)

    # ── §9: Penalty ───────────────────────────────────────────────────
    penalty              = _stuffing_penalty(resume_text, jd_skills)
    breakdown['penalty'] = round(penalty, 2)

    # ── Final score ───────────────────────────────────────────────────
    # Stuffing guard: when severe stuffing detected, halve keyword/skill scores
    if penalty < -6:
        breakdown['keyword'] = round(breakdown['keyword'] * 0.50, 2)
        breakdown['skills']  = round(breakdown['skills']  * 0.50, 2)

    # S-curve calibration
    factor_sum = sum(v for k, v in breakdown.items() if k != 'penalty')
    raw_total  = factor_sum + penalty

    if raw_total <= 0:
        final_score = 3.0
    elif raw_total < 20:
        # Weak-resume zone: map [0, 20] → [3, 20] with slight floor lift
        final_score = 3.0 + (raw_total / 20.0) * 17.0
    elif raw_total <= 80:
        # Linear zone — most real resumes live here
        final_score = raw_total
    else:
        # Ceiling zone: log-stretch [80,100] → [80,97]
        # Perfect resume needs ~94+ raw to reach 95 final
        excess      = raw_total - 80.0
        final_score = 80.0 + math.log1p(excess) / math.log1p(20.0) * 17.0

    final_score = round(min(100.0, max(0.0, final_score)), 1)

    # ── §11: Suggestions ──────────────────────────────────────────────
    suggestions = _build_suggestions(
        score         = final_score,
        missing       = sk_result.missing,
        breakdown     = breakdown,
        metrics_found = metrics_found,
        verb_count    = n_verbs,
        wc            = wc,
        sections      = sections,
        jd_text       = jd_text,
        domain        = domain,
        jd_skills     = jd_skills,
        jd_tools      = jd_tools,
        resume_lower  = resume_lower,
    )

    return {
        'score':          final_score,
        'missing_skills': sk_result.missing,
        'strengths':      sk_result.matched,
        'suggestions':    suggestions,
        'breakdown':      breakdown,
    }
