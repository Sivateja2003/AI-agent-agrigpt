import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

PESTS_KNOWLEDGE_BASE = """
Rice
Pests:
- Stem Borer (Scirpophaga incertulas) — Causes 'dead heart' in vegetative stage and 'white ear' at heading. Larvae bore into stems. Control: carbofuran 3G @ 25 kg/ha, light traps, release of Trichogramma parasitoids.
- Brown Planthopper (Nilaparvata lugens) — Sucks phloem sap causing 'hopper burn' — circular patches of dried plants. Control: Avoid excessive nitrogen, buprofezin 25 SC @ 1 ml/L, preserve natural enemies.
- Gall Midge (Orseolia oryzae) — Converts tillers into tubular galls called 'silver shoots'. Control: carbofuran 3G @ 25 kg/ha during tillering; resistant varieties Jaya, Vikram.
- Leaf Folder (Cnaphalocrocis medinalis) — Folds leaves longitudinally and feeds inside. Control: chlorpyrifos 20 EC @ 2 ml/L.
Diseases:
- Blast (Magnaporthe oryzae) — Diamond-shaped lesions with grey centers on leaves and neck. Control: tricyclazole 75 WP @ 0.6 g/L, use resistant varieties.
- Bacterial Leaf Blight (Xanthomonas oryzae) — Water-soaked to yellow lesions along leaf margins. Control: copper oxychloride 50 WP @ 3 g/L, avoid flood irrigation.
- Sheath Blight (Rhizoctonia solani) — Oval lesions on sheath with whitish center and brown border. Control: hexaconazole 5 EC @ 2 ml/L.
- Brown Spot (Helminthosporium oryzae) — Circular to oval brown spots on leaves. Control: mancozeb 75 WP @ 2.5 g/L, balanced fertilization.

Wheat
Pests:
- Aphids (Sitobion avenae) — Cluster on leaves and heads, cause yellowing and sooty mold. Control: imidacloprid 17.8 SL @ 0.5 ml/L.
- Termites (Odontotermes spp.) — Feed on roots causing wilting. Control: chlorpyrifos 20 EC @ 4 L/ha in irrigation water.
Diseases:
- Yellow Rust / Stripe Rust (Puccinia striiformis) — Yellow stripes of pustules on leaves. Control: propiconazole 25 EC @ 1 ml/L at first appearance.
- Brown Rust (Puccinia recondita) — Orange-brown pustules scattered on leaves. Control: mancozeb 75 WP @ 2.5 g/L.
- Loose Smut (Ustilago tritici) — Entire ear replaced by black smut mass. Control: seed treatment with carboxin 75 WP @ 2 g/kg seed.
- Karnal Bunt (Tilletia indica) — Partial conversion of grain to black powder. Control: seed treatment with thiram 75 WP @ 2.5 g/kg.

Tomato
Pests:
- Fruit Borer (Helicoverpa armigera) — Larvae bore into fruits leaving circular holes with frass. Control: spinosad 45 SC @ 0.3 ml/L, pheromone traps.
- Whitefly (Bemisia tabaci) — Transmits leaf curl virus; adults and nymphs suck sap from underside of leaves. Control: imidacloprid 17.8 SL @ 0.5 ml/L, yellow sticky traps.
- Leaf Miner (Liriomyza trifolii) — Serpentine mines on leaves. Control: abamectin 1.9 EC @ 1 ml/L.
Diseases:
- Early Blight (Alternaria solani) — Dark brown spots with concentric rings (target board appearance) on older leaves. Control: mancozeb 75 WP @ 2.5 g/L every 7-10 days.
- Late Blight (Phytophthora infestans) — Irregular water-soaked lesions turning brown on leaves and fruits. Control: metalaxyl + mancozeb @ 2.5 g/L, avoid overhead irrigation.
- Leaf Curl Virus (TYLCV) — Upward curling, yellowing, and stunting of leaves. Transmitted by whitefly. Control: vector control, resistant varieties, remove infected plants.
- Fusarium Wilt (Fusarium oxysporum) — Yellowing from lower leaves upward, vascular browning. Control: carbendazim soil treatment, grafted plants, crop rotation.

Cotton
Pests:
- Pink Bollworm (Pectinophora gossypiella) — Larvae bore into bolls; 'rosette flowers' are a sign of infestation. Control: pheromone traps @ 5/ha, spinosad 45 SC @ 0.3 ml/L, Bt cotton varieties.
- American Bollworm (Helicoverpa armigera) — Larvae feed on squares, flowers, and bolls. Control: indoxacarb 14.5 SC @ 1 ml/L, HaNPV @ 250 LE/ha.
- Sucking Pests (Aphids, Jassids, Thrips, Whitefly) — Cause leaf curling, yellowing, and honeydew deposits. Control: thiamethoxam 25 WG @ 0.3 g/L.
- Mealy Bug (Phenacoccus solenopsis) — White cottony masses on stems; wilting and drying of plants. Control: profenofos 50 EC @ 2 ml/L.
Diseases:
- Cotton Leaf Curl Disease (CLCuD) — Upward or downward curling of leaves, thickening of veins, enations. Spread by whitefly. Control: vector control, remove infected plants, tolerant varieties.
- Bacterial Blight (Xanthomonas axonopodis) — Angular water-soaked spots on cotyledons and leaves. Control: streptocycline seed treatment, copper oxychloride 50 WP @ 3 g/L.

Maize
Pests:
- Fall Armyworm (Spodoptera frugiperda) — Feeds on whorl leaves leaving ragged holes and frass. Control: chlorpyrifos 20 EC @ 2 ml/L in whorl, Bt-based biopesticides.
- Stem Borer (Chilo partellus) — Dead heart at vegetative stage; tunnels in stem. Control: carbofuran 3G @ 20 kg/ha in whorl.
Diseases:
- Turcicum Leaf Blight (Exserohilum turcicum) — Long elliptical grey-green to tan lesions. Control: mancozeb 75 WP @ 2.5 g/L, resistant hybrids.
- Maydis Leaf Blight (Helminthosporium maydis) — Yellowish or tan lesions parallel to leaf margins. Control: zineb 75 WP @ 2 g/L.

General IPM Tips:
- Integrated Pest Management (IPM): Combine cultural, biological, and chemical methods. Prefer bio-pesticides first.
- Crop Rotation: Breaks pest and disease cycles. Avoid growing the same crop family consecutively.
- Seed Treatment: Treat seeds with fungicide/insecticide before sowing to protect against soil-borne diseases and early pests.
- Pheromone Traps: Use species-specific traps at 5 traps/ha to monitor and mass-trap bollworms and fruit borers.
- Light Traps: Install 1 trap/ha to attract and kill nocturnal insects like stem borers and armyworms.
- Neem-based Products: Neem oil @ 3 ml/L or NSKE 5% acts as a repellent and growth inhibitor for many pests.
- Resistant Varieties: Always prefer crop varieties resistant or tolerant to locally prevalent pests and diseases.
"""

SCHEMES_KNOWLEDGE_BASE = """
PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)
Benefit: Rs 6,000 per year paid in 3 instalments of Rs 2,000 directly to farmer's bank account.
Eligibility: All small and marginal landholding farmer families across India.
Exclusions: Income tax payers, institutional land holders, government employees, constitutional post holders.
How to apply: Visit pmkisan.gov.in or nearest Common Service Centre (CSC). Submit Aadhaar + land records + bank account.
Helpline: 155261 / 011-24300606. Portal: pmkisan.gov.in

PM Fasal Bima Yojana (PMFBY) — Crop Insurance Scheme
Covers: Crop loss due to natural calamities, pests, diseases, post-harvest losses.
Farmer premium: Kharif crops — 2% of sum insured; Rabi crops — 1.5%; Commercial/Horticulture — 5%.
Remaining premium shared equally by state and central government.
How to apply: Through bank (automatic if crop loan taken), CSC, or insurance company before sowing cut-off date.
Helpline: 14447 / 1800-200-7710

Kisan Credit Card (KCC)
Purpose: Short-term credit for crop cultivation, post-harvest expenses, and allied activities (fisheries, animal husbandry).
Credit limit: Typically Rs 1.60 lakh without collateral, based on land holding, crop cultivated, and scale of finance.
Interest rate: 7% per annum (effective 4% with interest subvention if repaid on time).
Validity: 5 years (renewable).
How to apply: Visit nearest bank branch — all nationalised banks, RRBs, cooperative banks.
Documents needed: Aadhaar card, land ownership/lease documents, passport photo, bank account details.

Soil Health Card Scheme
Purpose: Provides farmers a card with details of soil nutrient status and crop-wise fertilizer recommendations.
Benefit: Helps reduce input costs and improve yields through correct fertilizer application.
Frequency: Issued every 2 years.
How to get: Contact local Agriculture Department, Krishi Vigyan Kendra (KVK), or apply at soilhealth.dac.gov.in.
Contains: pH, EC, organic carbon, N/P/K, S, Zn, Fe, Mn, Cu, B levels and crop-wise fertilizer recommendations.

PM Kisan Maan Dhan Yojana — Farmer Pension Scheme
Benefit: Rs 3,000 per month pension after age 60.
Eligibility: Age 18-40 years, land holding up to 2 hectares, not covered under other pension schemes.
Contribution: Rs 55 to Rs 200 per month (based on age of entry), matched equally by Central Government.
How to apply: Visit nearest CSC with Aadhaar and Kisan passbook/land records.

NMSA — National Mission for Sustainable Agriculture
Focus: Soil health, water use efficiency, and climate-resilient farming.
Sub-scheme — Paramparagat Krishi Vikas Yojana (PKVY): Rs 50,000/ha over 3 years for organic farming clusters.
How to join: Through State Agriculture Department or FPO (Farmer Producer Organization).

Per Drop More Crop (Micro Irrigation) — Under PMKSY
Benefit: 55% subsidy on drip/sprinkler irrigation systems for small and marginal farmers; 45% for other farmers.
Purpose: Reduce water usage and increase crop yield per drop of water.
How to apply: Through State Horticulture or Agriculture Department, or PMKSY portal.

eNAM — National Agriculture Market
Purpose: Online trading platform connecting farmers directly to buyers, traders, and exporters.
Benefit: Transparent price discovery, better prices, no middlemen.
How to register: Contact nearest APMC (Mandi). Register with Aadhaar + bank account + land/produce details.
Portal: enam.gov.in

RKVY — Rashtriya Krishi Vikas Yojana
Purpose: Overall agricultural development — infrastructure, equipment, technology.
Benefit: Subsidies on farm machinery, cold storage, post-harvest infrastructure, and value addition units.
How to access: Apply through State Agriculture Department or District Agriculture Officer.

Agri Infrastructure Fund
Purpose: Finance for post-harvest management and agri-logistics infrastructure.
Benefit: Loans up to Rs 2 crore with 3% interest subvention and credit guarantee.
Eligible projects: Cold storage, warehouses, sorting/grading units.
Eligibility: Farmers, FPOs, SHGs, agri-startups.
How to apply: Through banks or agriinfra.dac.gov.in. Portal: agriinfra.dac.gov.in

Key Helplines & Resources:
- Kisan Call Centre: 1800-180-1551 (toll-free, 24x7, available in local languages)
- PM-KISAN Helpline: 155261
- Crop Insurance Helpline: 14447
- One-stop portal for all schemes: farmer.gov.in
"""

_PEST_SYSTEM = (
    "You are an expert agricultural entomologist and pest management specialist for Indian farmers. "
    "Use the following knowledge base as your primary reference:\n\n"
    + PESTS_KNOWLEDGE_BASE
    + "\n\nProvide detailed, practical information covering: pest identification, damage & symptoms, "
    "affected crops, prevention, organic control, and chemical control with safety precautions."
)

_SCHEME_SYSTEM = (
    "You are an expert on Indian government agricultural schemes and farmer support programs. "
    "Use the following knowledge base as your primary reference:\n\n"
    + SCHEMES_KNOWLEDGE_BASE
    + "\n\nStructure your response covering: Scheme Overview, Eligibility, Benefits/Amount, "
    "How to Apply, Required Documents, Deadlines, and Helpline/Contact."
)


def _make_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)


def run_pest_tool(api_key: str, query: str) -> str:
    llm = _make_llm(api_key)
    return llm.invoke([SystemMessage(content=_PEST_SYSTEM), HumanMessage(content=query)]).content


def run_scheme_tool(api_key: str, query: str) -> str:
    llm = _make_llm(api_key)
    return llm.invoke([SystemMessage(content=_SCHEME_SYSTEM), HumanMessage(content=query)]).content


def query_pest_structured(api_key: str, query: str) -> dict:
    """Used by /pests endpoint — returns {pest, details}."""
    system = (
        _PEST_SYSTEM
        + "\n\nRespond ONLY with a JSON object in this exact format: "
        "{\"pest\": \"<pest or disease name>\", \"details\": \"<detailed info including symptoms and control measures>\"}. "
        "No markdown fences, no extra text — only the JSON object."
    )
    llm = _make_llm(api_key)
    text = _strip_json_fences(
        llm.invoke([SystemMessage(content=system), HumanMessage(content=query)]).content.strip()
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"pest": "Unknown", "details": text}


def query_scheme_structured(api_key: str, query: str) -> dict:
    """Used by /schemes endpoint — returns {scheme, details}."""
    system = (
        _SCHEME_SYSTEM
        + "\n\nRespond ONLY with a JSON object in this exact format: "
        "{\"scheme\": \"<scheme name>\", \"details\": \"<detailed info about the scheme>\"}. "
        "No markdown fences, no extra text — only the JSON object."
    )
    llm = _make_llm(api_key)
    text = _strip_json_fences(
        llm.invoke([SystemMessage(content=system), HumanMessage(content=query)]).content.strip()
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"scheme": "Unknown", "details": text}


def _strip_json_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
