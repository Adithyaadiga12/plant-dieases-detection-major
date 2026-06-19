import '../models/disease_info.dart';

const DiseaseInfo unknownDisease = DiseaseInfo(
  name: 'Unknown',
  description: 'The image could not be matched to a known plant disease.',
  symptoms: 'No identifiable symptoms from the given image.',
  causes: 'Image quality or plant type may not match training data.',
  prevention: 'Use clean tools, avoid wet foliage, monitor plants regularly.',
  remedy: 'Retake with better lighting, or consult an agriculture expert.',
  isHealthy: false,
);

// Keyed by the RAW class name from class_names.dart
const Map<String, DiseaseInfo> _diseaseDb = {
  // ── Apple ──────────────────────────────────────────────────────────────────
  'Apple___Apple_scab': DiseaseInfo(
    name: 'Apple Scab',
    description:
        'A fungal disease caused by Venturia inaequalis that affects apple leaves, fruit, and shoots.',
    symptoms:
        'Olive-green or brown velvety spots on leaves, scabby crust on fruit surface, premature leaf drop.',
    causes:
        'Cool, wet spring weather favors spore release and infection of young tissue.',
    prevention:
        'Plant resistant varieties, rake and dispose of fallen leaves, maintain canopy airflow.',
    remedy:
        'Apply fungicides (captan, myclobutanil) at early bud stage; repeat every 7–10 days during wet periods.',
    isHealthy: false,
  ),
  'Apple___Black_rot': DiseaseInfo(
    name: 'Apple Black Rot',
    description:
        'A fungal disease caused by Botryosphaeria obtusa affecting leaves, fruit, and bark.',
    symptoms:
        'Purple-bordered leaf spots turning brown, mummified fruit with concentric rings, cankers on bark.',
    causes:
        'Warm, humid conditions; spread through infected wood and mummified fruit.',
    prevention:
        'Prune dead wood, remove mummified fruit, keep orchard clean of debris.',
    remedy:
        'Apply thiophanate-methyl or captan fungicide; prune cankers to healthy wood.',
    isHealthy: false,
  ),
  'Apple___Cedar_apple_rust': DiseaseInfo(
    name: 'Cedar Apple Rust',
    description:
        'A fungal disease caused by Gymnosporangium juniperi-virginianae requiring both apple and cedar host plants.',
    symptoms:
        'Bright orange-yellow spots on upper leaf surface, tube-like spore projections beneath leaves.',
    causes:
        'Spores spread from nearby cedar/juniper trees during spring wet periods.',
    prevention:
        'Remove cedar trees within 1–2 km, plant resistant apple varieties.',
    remedy:
        'Apply fungicides (myclobutanil, triadimefon) just before and after bloom.',
    isHealthy: false,
  ),
  'Apple___healthy': DiseaseInfo(
    name: 'Apple Healthy',
    description: 'The apple leaf shows no signs of disease.',
    symptoms: 'Uniform green color, no spots or lesions, intact leaf surface.',
    causes: 'Good plant health with no detected pathogens.',
    prevention: 'Continue regular watering, balanced fertilization, and pruning.',
    remedy: 'No treatment needed. Maintain current care routine.',
    isHealthy: true,
  ),
  // ── Blueberry ──────────────────────────────────────────────────────────────
  'Blueberry___healthy': DiseaseInfo(
    name: 'Blueberry Healthy',
    description: 'The blueberry leaf is healthy with no disease signs.',
    symptoms: 'Vibrant green leaves, no discoloration or spots.',
    causes: 'Healthy growing conditions.',
    prevention: 'Maintain soil pH 4.5–5.5, ensure adequate drainage.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Cherry ─────────────────────────────────────────────────────────────────
  'Cherry_(including_sour)___Powdery_mildew': DiseaseInfo(
    name: 'Cherry Powdery Mildew',
    description:
        'A fungal disease caused by Podosphaera clandestina that coats cherry leaves in white powder.',
    symptoms:
        'White powdery coating on young leaves and shoots, leaf curling and distortion, stunted growth.',
    causes:
        'Warm dry days with cool humid nights; spread by wind-borne spores.',
    prevention:
        'Avoid overcrowding, improve air circulation, avoid excess nitrogen.',
    remedy:
        'Apply sulfur-based or potassium bicarbonate fungicide; neem oil as organic option.',
    isHealthy: false,
  ),
  'Cherry_(including_sour)___healthy': DiseaseInfo(
    name: 'Cherry Healthy',
    description: 'The cherry leaf is healthy.',
    symptoms: 'Glossy green leaves, no powder or spots.',
    causes: 'Good growing conditions.',
    prevention: 'Ensure full sun, adequate spacing, and annual pruning.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Corn ───────────────────────────────────────────────────────────────────
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': DiseaseInfo(
    name: 'Corn Gray Leaf Spot',
    description:
        'A fungal disease caused by Cercospora zeae-maydis, one of the most yield-limiting corn diseases.',
    symptoms:
        'Rectangular, tan to gray lesions with sharp edges running parallel to leaf veins.',
    causes:
        'High humidity, minimal air movement, and dense canopies favor infection.',
    prevention:
        'Use resistant hybrids, rotate crops with non-corn species, till residue.',
    remedy:
        'Apply strobilurin or triazole fungicides at tasseling if disease is present.',
    isHealthy: false,
  ),
  'Corn_(maize)___Common_rust_': DiseaseInfo(
    name: 'Corn Common Rust',
    description:
        'A fungal disease caused by Puccinia sorghi, common in cool humid corn-growing regions.',
    symptoms:
        'Small, oval to elongated brick-red pustules on both leaf surfaces, turning dark brown at maturity.',
    causes:
        'Cool temperatures (16–23°C) with high humidity and dew; spores carried by wind.',
    prevention:
        'Plant resistant hybrids, monitor fields from mid-season onward.',
    remedy:
        'Apply fungicide (propiconazole, azoxystrobin) when disease first appears on upper leaves.',
    isHealthy: false,
  ),
  'Corn_(maize)___Northern_Leaf_Blight': DiseaseInfo(
    name: 'Corn Northern Leaf Blight',
    description:
        'A fungal disease caused by Exserohilum turcicum that damages corn leaves and reduces yield.',
    symptoms:
        'Large, cigar-shaped tan lesions (2.5–15 cm), with wavy edges on leaves, starting on lower canopy.',
    causes:
        'Moderate temperatures and prolonged leaf wetness promote infection and spore production.',
    prevention:
        'Use resistant varieties, crop rotation, and bury or till infected residue.',
    remedy:
        'Apply foliar fungicide at the appearance of first lesions, especially near tasseling.',
    isHealthy: false,
  ),
  'Corn_(maize)___healthy': DiseaseInfo(
    name: 'Corn Healthy',
    description: 'The corn leaf shows no disease.',
    symptoms: 'Broad, dark green leaves with no lesions or discoloration.',
    causes: 'Healthy growing environment.',
    prevention: 'Adequate irrigation, balanced fertilization, and weed control.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Grape ──────────────────────────────────────────────────────────────────
  'Grape___Black_rot': DiseaseInfo(
    name: 'Grape Black Rot',
    description:
        'A fungal disease caused by Guignardia bidwellii that destroys grape berries and foliage.',
    symptoms:
        'Reddish-brown circular leaf spots, berries turn brown then black and shriveled (mummies).',
    causes:
        'Warm wet weather (24–29°C) during shoot growth, bloom, and fruit development.',
    prevention:
        'Remove mummified berries and dead canes, improve canopy airflow.',
    remedy:
        'Apply mancozeb or myclobutanil from bud break through berry set; remove infected material.',
    isHealthy: false,
  ),
  'Grape___Esca_(Black_Measles)': DiseaseInfo(
    name: 'Grape Esca (Black Measles)',
    description:
        'A complex wood disease caused by multiple fungi affecting the vascular system of grapevines.',
    symptoms:
        'Interveinal chlorosis with brown stripes on leaves ("tiger stripe"), shrunken dark fruit with purple spots.',
    causes:
        'Combination of Phaeomoniella chlamydospora and wood-decaying fungi; often enters through pruning wounds.',
    prevention:
        'Use clean pruning tools, apply wound sealant, avoid late pruning in wet conditions.',
    remedy:
        'No effective cure; remove severely infected vines, manage stress, and protect wounds.',
    isHealthy: false,
  ),
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': DiseaseInfo(
    name: 'Grape Leaf Blight',
    description:
        'A fungal disease caused by Pseudocercospora vitis causing defoliation in vineyards.',
    symptoms:
        'Irregular brown or reddish-brown angular spots on upper leaf surface, white sporulation beneath.',
    causes:
        'High temperature and humidity; spread by rain-splashed spores.',
    prevention:
        'Proper vine spacing, remove infected leaves, and improve air circulation.',
    remedy:
        'Apply copper-based or mancozeb fungicide during early stages of infection.',
    isHealthy: false,
  ),
  'Grape___healthy': DiseaseInfo(
    name: 'Grape Healthy',
    description: 'The grape leaf is healthy.',
    symptoms: 'Uniform green leaves with no spots or wilting.',
    causes: 'Good vineyard management.',
    prevention: 'Regular pruning, proper irrigation, and canopy management.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Orange ─────────────────────────────────────────────────────────────────
  'Orange___Haunglongbing_(Citrus_greening)': DiseaseInfo(
    name: 'Citrus Greening (HLB)',
    description:
        'A devastating bacterial disease caused by Candidatus Liberibacter spp., spread by the Asian citrus psyllid. No cure exists.',
    symptoms:
        'Asymmetric mottling (blotchy yellowing), small lopsided bitter fruit, twig dieback, and overall tree decline.',
    causes:
        'Bacterial infection spread by Diaphorina citri (Asian citrus psyllid) insects.',
    prevention:
        'Control psyllid populations with insecticides, use certified disease-free nursery stock, quarantine infected areas.',
    remedy:
        'No cure — infected trees must be removed. Manage psyllid vectors aggressively to slow spread.',
    isHealthy: false,
  ),
  // ── Peach ──────────────────────────────────────────────────────────────────
  'Peach___Bacterial_spot': DiseaseInfo(
    name: 'Peach Bacterial Spot',
    description:
        'A bacterial disease caused by Xanthomonas arboricola pv. pruni affecting peach leaves and fruit.',
    symptoms:
        'Water-soaked angular spots on leaves turning brown and dropping out ("shot-hole"), sunken cracked lesions on fruit.',
    causes:
        'Warm humid weather and rain; bacteria overwinter in infected buds and twigs.',
    prevention:
        'Plant resistant varieties, avoid wounding, ensure good airflow in canopy.',
    remedy:
        'Apply copper-based bactericides preventively; remove and destroy infected material.',
    isHealthy: false,
  ),
  'Peach___healthy': DiseaseInfo(
    name: 'Peach Healthy',
    description: 'The peach leaf shows no signs of disease.',
    symptoms: 'Lush green leaves with no spots or lesions.',
    causes: 'Healthy growing conditions.',
    prevention: 'Regular thinning, fertilization, and pest monitoring.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Pepper ─────────────────────────────────────────────────────────────────
  'Pepper,_bell___Bacterial_spot': DiseaseInfo(
    name: 'Pepper Bacterial Spot',
    description:
        'A bacterial disease caused by Xanthomonas spp. affecting pepper leaves, stems, and fruit.',
    symptoms:
        'Small water-soaked spots on leaves turning yellow then brown with dark centers; raised scabby lesions on fruit.',
    causes:
        'Warm (24–30°C) wet conditions; spread by rain splash, insects, and contaminated tools.',
    prevention:
        'Use disease-free seed, avoid overhead irrigation, rotate crops annually.',
    remedy:
        'Apply copper bactericide; remove infected plant debris and avoid working in wet fields.',
    isHealthy: false,
  ),
  'Pepper,_bell___healthy': DiseaseInfo(
    name: 'Pepper Healthy',
    description: 'The bell pepper leaf is healthy.',
    symptoms: 'Deep green glossy leaves, no spots or discoloration.',
    causes: 'Optimal growing conditions.',
    prevention: 'Drip irrigation, balanced fertilization, crop rotation.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Potato ─────────────────────────────────────────────────────────────────
  'Potato___Early_blight': DiseaseInfo(
    name: 'Potato Early Blight',
    description:
        'A fungal disease caused by Alternaria solani, common in warm humid potato-growing regions.',
    symptoms:
        'Dark brown concentric ring spots (bull\'s-eye pattern) on older leaves; yellow halo around lesions.',
    causes:
        'Warm temperatures (24–29°C), high humidity, and stress from nutrient deficiency or drought.',
    prevention:
        'Use certified seed, crop rotation, adequate irrigation and nitrogen levels.',
    remedy:
        'Apply chlorothalonil or mancozeb fungicide; remove severely infected lower leaves.',
    isHealthy: false,
  ),
  'Potato___Late_blight': DiseaseInfo(
    name: 'Potato Late Blight',
    description:
        'A water mold disease caused by Phytophthora infestans — the pathogen that caused the Irish famine.',
    symptoms:
        'Dark water-soaked lesions rapidly covering leaves, white fuzzy growth on leaf undersides, rapid plant collapse.',
    causes:
        'Cool (10–20°C), wet conditions with high humidity and fog; extremely fast spreading.',
    prevention:
        'Plant resistant varieties, avoid overhead watering, destroy volunteer potato plants.',
    remedy:
        'Apply metalaxyl-M, cymoxanil, or copper fungicides; act immediately on first symptoms.',
    isHealthy: false,
  ),
  'Potato___healthy': DiseaseInfo(
    name: 'Potato Healthy',
    description: 'The potato leaf is healthy.',
    symptoms: 'Uniform dark green leaves with no lesions.',
    causes: 'Good soil health and growing conditions.',
    prevention: 'Use certified seed potatoes, proper hilling, balanced fertilization.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Raspberry ──────────────────────────────────────────────────────────────
  'Raspberry___healthy': DiseaseInfo(
    name: 'Raspberry Healthy',
    description: 'The raspberry leaf is healthy.',
    symptoms: 'Bright green serrated leaves with no disease signs.',
    causes: 'Good growing conditions and management.',
    prevention: 'Prune annually, provide support structures, maintain drainage.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Soybean ────────────────────────────────────────────────────────────────
  'Soybean___healthy': DiseaseInfo(
    name: 'Soybean Healthy',
    description: 'The soybean leaf is healthy.',
    symptoms: 'Trifoliate leaves with no yellowing, spots, or wilting.',
    causes: 'Optimal growing environment.',
    prevention: 'Crop rotation, balanced fertilization, adequate plant spacing.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Squash ─────────────────────────────────────────────────────────────────
  'Squash___Powdery_mildew': DiseaseInfo(
    name: 'Squash Powdery Mildew',
    description:
        'A fungal disease caused by Podosphaera xanthii and Erysiphe cichoracearum affecting squash and cucurbits.',
    symptoms:
        'White powdery spots on upper and lower leaf surfaces, leaves yellow and wither, reduced fruit quality.',
    causes:
        'Warm dry days with high humidity at night; spread by wind-borne spores.',
    prevention:
        'Plant resistant varieties, avoid overhead watering, improve air circulation.',
    remedy:
        'Apply potassium bicarbonate, neem oil, or sulfur fungicide; remove heavily infected leaves.',
    isHealthy: false,
  ),
  // ── Strawberry ─────────────────────────────────────────────────────────────
  'Strawberry___Leaf_scorch': DiseaseInfo(
    name: 'Strawberry Leaf Scorch',
    description:
        'A fungal disease caused by Diplocarpon earlianum causing severe defoliation in strawberry.',
    symptoms:
        'Small dark purple spots on upper leaf surface, spots enlarge and coalesce causing leaf to look scorched.',
    causes:
        'Warm humid weather; survives on infected plant debris and spreads by rain splash.',
    prevention:
        'Remove old foliage after harvest, avoid overcrowding, use drip irrigation.',
    remedy:
        'Apply copper or captan-based fungicide; mow and remove infected leaves post-harvest.',
    isHealthy: false,
  ),
  'Strawberry___healthy': DiseaseInfo(
    name: 'Strawberry Healthy',
    description: 'The strawberry leaf is healthy.',
    symptoms: 'Bright green trifoliate leaves with no spots or discoloration.',
    causes: 'Good growing conditions.',
    prevention: 'Proper spacing, drip irrigation, regular renovation of beds.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
  // ── Tomato ─────────────────────────────────────────────────────────────────
  'Tomato___Bacterial_spot': DiseaseInfo(
    name: 'Tomato Bacterial Spot',
    description:
        'A bacterial disease caused by Xanthomonas spp. affecting tomato leaves, stems, and fruit.',
    symptoms:
        'Small water-soaked lesions on leaves turning brown with yellow halo; dark raised spots on fruit.',
    causes:
        'Warm wet conditions; spread by rain splash, contaminated seeds, and equipment.',
    prevention:
        'Use certified disease-free seeds, avoid overhead irrigation, rotate crops.',
    remedy:
        'Apply copper-based bactericide; remove infected material and avoid working when wet.',
    isHealthy: false,
  ),
  'Tomato___Early_blight': DiseaseInfo(
    name: 'Tomato Early Blight',
    description:
        'A common fungal disease caused by Alternaria solani affecting tomato leaves, stems, and fruit.',
    symptoms:
        'Brown circular spots with concentric rings (target pattern), yellow halo, starting on older lower leaves.',
    causes:
        'Warm humid weather, infected plant debris, poor air circulation.',
    prevention:
        'Remove infected leaves, crop rotation, avoid overhead watering, mulch soil.',
    remedy:
        'Apply copper-based or chlorothalonil fungicide; improve plant spacing for airflow.',
    isHealthy: false,
  ),
  'Tomato___Late_blight': DiseaseInfo(
    name: 'Tomato Late Blight',
    description:
        'A devastating disease caused by Phytophthora infestans that can wipe out an entire crop rapidly.',
    symptoms:
        'Dark water-soaked patches on leaves, white mold on leaf undersides in humid conditions, rapid plant collapse.',
    causes:
        'Cool (10–20°C) and wet conditions; spreads by wind and rain.',
    prevention:
        'Plant resistant varieties, avoid overhead watering, destroy infected plants immediately.',
    remedy:
        'Apply metalaxyl or copper fungicide at first sign; remove and destroy all infected plants.',
    isHealthy: false,
  ),
  'Tomato___Leaf_Mold': DiseaseInfo(
    name: 'Tomato Leaf Mold',
    description:
        'A fungal disease caused by Passalora fulva, most severe in greenhouses and high-humidity environments.',
    symptoms:
        'Pale green to yellow spots on upper leaf surface; olive-green to brown velvety mold on undersides.',
    causes:
        'High humidity (>85%) and moderate temperatures; poor ventilation in greenhouses.',
    prevention:
        'Improve greenhouse ventilation, reduce humidity, avoid wetting leaves.',
    remedy:
        'Apply fungicide (chlorothalonil, mancozeb); remove infected leaves and improve airflow.',
    isHealthy: false,
  ),
  'Tomato___Septoria_leaf_spot': DiseaseInfo(
    name: 'Tomato Septoria Leaf Spot',
    description:
        'A fungal disease caused by Septoria lycopersici — one of the most destructive tomato leaf diseases.',
    symptoms:
        'Small circular spots (3–5 mm) with dark borders and gray-white centers containing small black dots.',
    causes:
        'Warm humid conditions; spread by rain splash from infected soil and plant debris.',
    prevention:
        'Mulch soil, remove lower infected leaves, avoid overhead irrigation.',
    remedy:
        'Apply fungicide (chlorothalonil, copper) and remove diseased foliage early.',
    isHealthy: false,
  ),
  'Tomato___Spider_mites Two-spotted_spider_mite': DiseaseInfo(
    name: 'Tomato Spider Mites',
    description:
        'An infestation by Tetranychus urticae (two-spotted spider mite) that feeds on tomato leaf cells.',
    symptoms:
        'Yellow stippling on leaves, fine webbing on underside, bronzed leaf appearance, leaf drop in severe cases.',
    causes:
        'Hot dry conditions, dusty environment; populations explode rapidly in summer.',
    prevention:
        'Avoid excessive nitrogen, keep plants well-watered, use predatory mites.',
    remedy:
        'Apply miticide (abamectin, spiromesifen) or neem oil; increase humidity around plants.',
    isHealthy: false,
  ),
  'Tomato___Target_Spot': DiseaseInfo(
    name: 'Tomato Target Spot',
    description:
        'A fungal disease caused by Corynespora cassiicola affecting tomato leaves, stems, and fruit.',
    symptoms:
        'Brown spots with concentric rings and yellow halo on leaves; dark sunken lesions on stems and fruit.',
    causes:
        'Warm humid conditions (24–32°C), prolonged leaf wetness, poor canopy airflow.',
    prevention:
        'Ensure adequate plant spacing, stake plants, avoid overhead watering.',
    remedy:
        'Apply azoxystrobin or chlorothalonil fungicide; remove and destroy infected plant parts.',
    isHealthy: false,
  ),
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': DiseaseInfo(
    name: 'Tomato Yellow Leaf Curl Virus',
    description:
        'A devastating viral disease transmitted by Bemisia tabaci (whitefly) with no chemical cure.',
    symptoms:
        'Upward leaf curling, yellowing of leaf margins, stunted growth, and severe yield loss.',
    causes:
        'Tomato yellow leaf curl virus (TYLCV) spread exclusively by silverleaf whitefly.',
    prevention:
        'Use resistant varieties, control whitefly with yellow sticky traps and insecticides, use reflective mulch.',
    remedy:
        'No cure for infected plants. Remove and destroy infected plants. Manage whitefly aggressively.',
    isHealthy: false,
  ),
  'Tomato___Tomato_mosaic_virus': DiseaseInfo(
    name: 'Tomato Mosaic Virus',
    description:
        'A highly contagious viral disease caused by Tomato mosaic virus (ToMV) that reduces yield and quality.',
    symptoms:
        'Mosaic (light/dark green mottling) pattern on leaves, leaf distortion, stunted growth, fruit bronzing.',
    causes:
        'Mechanically spread through contaminated hands, tools, and infected plant material; very stable virus.',
    prevention:
        'Use certified virus-free seeds, disinfect tools with bleach, wash hands, avoid tobacco near plants.',
    remedy:
        'No cure; remove infected plants. Control aphid vectors and sanitize all equipment.',
    isHealthy: false,
  ),
  'Tomato___healthy': DiseaseInfo(
    name: 'Tomato Healthy',
    description: 'The tomato leaf is healthy with no disease signs.',
    symptoms: 'Dark green leaves, no spots, curling, or discoloration.',
    causes: 'Healthy growing environment.',
    prevention: 'Regular watering, balanced fertilization, staking, and monitoring.',
    remedy: 'No action needed.',
    isHealthy: true,
  ),
};

DiseaseInfo diseaseInfoForRawClass(String rawClassName) {
  return _diseaseDb[rawClassName] ?? unknownDisease;
}

// For disease library browsing — all unique diseases excluding duplicates
List<MapEntry<String, DiseaseInfo>> get allDiseaseEntries =>
    _diseaseDb.entries.toList();
