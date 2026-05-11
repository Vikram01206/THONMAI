import { GoogleGenAI, Type, Modality } from "@google/genai";
import { assembleTamilText, RecognizedChar } from "../lib/tamilComposer";

let genAI: GoogleGenAI | null = null;

function getAI() {
  if (!genAI) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not defined. Please set it in your environment variables.");
    }
    genAI = new GoogleGenAI({ apiKey });
  }
  return genAI;
}

export type PipelineMode = 'literal' | 'historical';

export interface GlyphCandidate {
  glyph: string;
  predictions: Array<{
    char: string;
    probability: number;
    visualSimilarityScore: number; // Added to prioritize visual evidence
  }>;
}

export interface ReconstructionResult {
  modernTamil: string;
  historicalTamil: string;
  phoneticTransliteration: string; 
  confidence: number;
  grammarCorrectionNote: string;
  recoveredWordBoundaries: string[];
  compositionDebug?: string[];
  compositionWarnings?: string[];
}

export interface OCRResult {
  glyphs: string[];
  scriptFamily: string;
  detectedCentury: string;
  confidence: number;
  segments: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    glyph: string;
  }>;
}

export async function generateSpeech(text: string): Promise<string> {
  try {
    const response = await getAI().models.generateContent({
      model: "gemini-3.1-flash-tts-preview",
      contents: [{ parts: [{ text: `Say clearly in Tamil: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' }, // Kore is a high-quality female voice
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) throw new Error("No audio data returned");
    return base64Audio;
  } catch (error) {
    console.error("TTS Failed:", error);
    throw error;
  }
}

export async function performOCROnInscription(
  imageBase64: string,
  mimeType: string,
  mode: PipelineMode = 'historical'
): Promise<OCRResult> {
  const prompt = `
    You are a Research-Grade Epigraphic Transliteration Engine. 
    Analyze the provided ancient Tamil inscription image.

    CURRENT MODE: ${mode.toUpperCase()}
    
    TASK: Implement Adaptive Inscription Segmentation and Recognition.
    
    1. SEGMENTATION STRATEGY:
       - Apply strictly adaptive segmentation. 
       - Use Connected-Component Analysis (CCA) logic to isolate glyphs.
       - Use Vertical/Horizontal Projection Histograms to identify character boundaries.
       - PERFORM CONTOUR MERGE CORRECTION: If a contour spans multiple logical glyphs, split it based on historical width-to-height characteristics.
       - PERFORM OVERLAP SUPPRESSION: Ensure noise and stone texture artifacts (scratches, pits) are suppressed and not segmented as glyphs.
       - GLYPH BOUNDARY STABILIZATION: Ensure vowel markers and 'pulli' are attached to their parent consonant component and not segmented as independent noisy contours.

    2. RECOGNITION & VALIDATION:
       - SCRIPT CLASSIFICATION: Explicitly classify the glyphs as Tamil-Brahmi, Vatteluttu (Early/Middle/Late), Grantha-influenced Tamil, or Medieval Chola/Pandya Tamil. 
       - REJECT: Tiny noise contours, non-textual stone patterns, and low-density regions that do not form coherent character structures.
       - FIDELITY: Prioritize strict visual evidence. Do not hallucinate glyphs that are not physically detectable.
       - DAMAGED REGIONS: If a glyph is partially eroded, classify based on surviving stroke patterns only.

    3. OUTPUT:
       - Sort the segments in strict reading order (usually Top-to-Bottom, Left-to-Right for Tamil inscriptions).
       - Provide bounding boxes in normalized coordinates (0-1000).
       - Return full glyph sequence and confidence scores.
  `;

  try {
    const imagePart = {
      inlineData: {
        data: imageBase64,
        mimeType: mimeType,
      },
    };

    const response = await getAI().models.generateContent({
      model: "gemini-3-flash-preview",
      contents: { parts: [imagePart, { text: prompt }] },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            glyphs: { 
              type: Type.ARRAY, 
              items: { type: Type.STRING } 
            },
            scriptFamily: { type: Type.STRING },
            detectedCentury: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            segments: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  x: { type: Type.NUMBER },
                  y: { type: Type.NUMBER },
                  width: { type: Type.NUMBER },
                  height: { type: Type.NUMBER },
                  glyph: { type: Type.STRING },
                  confidence: { type: Type.NUMBER }
                },
                required: ["x", "y", "width", "height", "glyph"]
              }
            }
          },
          required: ["glyphs", "scriptFamily", "detectedCentury", "confidence", "segments"]
        },
      },
    });

    const parsed: OCRResult = JSON.parse(response.text);
    
    // Ensure segments are sorted by reading order (Y then X)
    parsed.segments.sort((a, b) => {
      if (Math.abs(a.y - b.y) < 20) return a.x - b.x;
      return a.y - b.y;
    });

    return parsed;
  } catch (error) {
    console.error("OCR Failed:", error);
    throw error;
  }
}

export async function reconstructHistorical(
  modernTamil: string,
  century: string
): Promise<string> {
  const prompt = `
    You are an expert in Tamil epigraphy and historical linguistics, specializing in the evolution of Tamil scripts.
    TASK: Reconstruct the historical script representation for the given modern Tamil text with extreme accuracy for the target period.

    TARGET PERIOD/ERA: "${century}"
    MODERN INPUT: "${modernTamil}"

    CONTEXTUAL GUIDELINES BASED ON PERIOD:
    1. TAMIL-BRAHMI (Approx. 300 BCE - 300 CE):
       - Use characters from the Tamil-Brahmi / Brahmi Unicode block.
       - Strictly omit 'pulli' (dots for consonants).
       - No word boundaries (continuous script).
       - Maintain "Mauna" (vowel-less) consonant clusters if applicable.

    2. EARLY VATTELUTTU (Approx. 400 CE - 800 CE):
       - Script is more rounded than Brahmi. 
       - If Unicode Vatteluttu is unavailable, use the specialized historical mapped characters.
       - Focus on the 'rounded' nature of letters like 'ta', 'pa', 'ma'.

    3. GRANTHA INFLUENCE / EARLY MEDIEVAL (Approx. 600 CE - 1000 CE):
       - Incorporate Sanskrit-derived Grantha characters for 'sha', 'ssa', 'ja', 'ha' if they appear in the input.
       - Script transition period between Vatteluttu and Medieval Tamil.

    4. IMPERIAL CHOLA (Approx. 1000 CE - 1300 CE):
       - More recognizable as ancestor to modern Tamil but with significant ornate variations.
       - Use 'pulli' selectively as per epigraphical evidence of that era.

    5. MODERN SCRIPT (Approx. 1500 CE+):
       - Standard modern Tamil orthography with fully evolved characters.

    CORE RULES:
    1. HISTORICAL ORTHOGRAPHY: Apply era-specific writing conventions strictly.
    2. SCRIPT REPRESENTATION: Return the text ONLY in the script of that era.
    3. NO EXPLANATIONS: Return only the reconstructed string.
    4. NO MODERN PUNCTUATION.

    HISTORICAL VALIDATION: Ensure the characters used are authentic to the epigraphical corpus of ${century}.
  `;

  try {
    const response = await getAI().models.generateContent({
      model: "gemini-3-flash-preview",
      contents: { parts: [{ text: prompt }] },
    });

    return response.text.trim();
  } catch (error) {
    console.error("Historical reconstruction failed:", error);
    throw error;
  }
}

export async function reconstructTamil(
  candidates: GlyphCandidate[],
  century: string,
  mode: PipelineMode = 'historical'
): Promise<ReconstructionResult> {
  const prompt = `
    You are an expert ancient Tamil Epigrapher. 
    TASK: Convert the detected glyph sequence into modern Tamil translitration. Mention the "${century}" context explicitly in the grammarCorrectionNote.

    CURRENT MODE: ${mode === 'literal' ? 'LITERAL TRANSLITERATION (PRIORITIZE VISUAL EVIDENCE)' : 'HISTORICAL RECONSTRUCTION (ALLOW CONTEXTUAL RESTORATION)'}

    INPUT GLYPHS:
    ${JSON.stringify(candidates, null, 2)}

    PERIOD: ${century}

    STRICT CONSTRAINTS (MANDATORY):
    1. VISUAL EVIDENCE PRIORITY: OCR visual identification probability MUST override language model probability. Do not "correct" a word if the visual evidence for the detected glyph contradicts the correction.
    2. PRESERVE GLYPH COUNT: The output modern Tamil string should strictly correspond to the number of detected glyph clusters in the input. Do not invent new syllables, titles, or prefixes.
    3. PREVENT HALLUCINATION: If a glyph is ambiguous, transliterate it as its top visual candidate (or "[?]" if mode is literal). Do not semantically complete sentences based on historical probability if the inscription does not show those characters.
    4. NO HONORIFIC EXPANSION: Do not automatically expand short names or titles into full honorific forms. Transliterate exactly what is written.
    5. BEAM SEARCH CONSTRAINTS: Limit contextual expansion. The result must stay structurally tied to the detected glyph sequence.

    MODE-SPECIFIC RULES:
    ${mode === 'literal' ? `
    - Mode: LITERAL
    - No semantic reconstruction.
    - No added syllables/vowels.
    - Output exactly what the glyph sequence indicates.
    ` : `
    - Mode: HISTORICAL
    - Controlled restoration allowed for damaged sections.
    - Contextual restoration of word boundaries and 'pulli'.
    - Use Sangam/Medieval linguistics only where structural evidence exists.
    `}

    Return the result as JSON.
  `;

  try {
    const response = await getAI().models.generateContent({
      model: "gemini-3-flash-preview",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            modernTamil: { type: Type.STRING },
            historicalTamil: { type: Type.STRING },
            phoneticTransliteration: { type: Type.STRING },
            confidence: { type: Type.NUMBER },
            grammarCorrectionNote: { type: Type.STRING },
            recoveredWordBoundaries: {
              type: Type.ARRAY,
              items: { type: Type.STRING }
            }
          },
          required: ["modernTamil", "historicalTamil", "phoneticTransliteration", "confidence", "grammarCorrectionNote", "recoveredWordBoundaries"]
        },
      },
    });

    const result = JSON.parse(response.text);

    // Apply Tamil Unicode Composition for proper orthography
    // Convert prediction candidates to RecognizedChar format for the composer
    const recognizedChars: RecognizedChar[] = candidates.map(c => ({
      character: c.predictions[0].char,
      confidence: c.predictions[0].probability,
    }));

    const composition = assembleTamilText(recognizedChars);
    
    // Use composed text if AI output is simple concatenation or looks fragmented
    // For now, we'll favor AI output but ensure it's at least NFC normalized
    result.modernTamil = result.modernTamil.normalize('NFC');
    result.compositionDebug = composition.debugSteps;
    result.compositionWarnings = composition.warnings;
    
    return result;
  } catch (error) {
    console.error("Tamil Reconstruction Failed:", error);
    
    // Fallback to the dedicated Tamil Composition Engine if AI fails
    const recognizedChars: RecognizedChar[] = candidates.map(c => ({
      character: c.predictions[0].char,
      confidence: c.predictions[0].probability,
    }));

    const composition = assembleTamilText(recognizedChars);

    return {
      modernTamil: composition.composedText,
      historicalTamil: composition.composedText, // Fallback best effort
      phoneticTransliteration: "",
      confidence: 0.5,
      grammarCorrectionNote: "AI processing failed, using deterministic Unicode Composition Engine.",
      recoveredWordBoundaries: [],
      compositionDebug: composition.debugSteps,
      compositionWarnings: composition.warnings
    };
  }
}
