import { GoogleGenAI, Type, Modality } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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
  phoneticTransliteration: string; // Added for pronunciation guide
  confidence: number;
  grammarCorrectionNote: string;
  recoveredWordBoundaries: string[];
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
    const response = await ai.models.generateContent({
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

    const response = await ai.models.generateContent({
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
    You are an expert in Tamil epigraphy and historical linguistics.
    TASK: Reconstruct the historical script representation for the given modern Tamil text.

    TARGET PERIOD: ${century}
    MODERN INPUT: "${modernTamil}"

    RULES:
    1. HISTORICAL ORTHOGRAPHY: Apply era-specific writing conventions. 
       - For early eras (Brahmi/Vatteluttu), omit word boundaries and consonant markers (pulli) where historically accurate.
       - For transitional eras (Pallava/Chola), use appropriate ligature forms.
    2. SCRIPT REPRESENTATION: Return the text as it would appear in the script of that era. 
       - Note: If Unicode doesn't fully support the specific ancient form, use the closest historical Unicode block (e.g., Tamil-Brahmi block or Vatteluttu-compatible Tamil characters).
    3. PHONETIC FIDELITY: Maintain the phonetic structure while adapting to historical script constraints.
    4. NO MODERN PUNCTUATION: Remove all modern commas, periods, or spaces if they didn't exist in the target era.

    Return only the reconstructed string.
  `;

  try {
    const response = await ai.models.generateContent({
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
    TASK: Convert the detected glyph sequence into modern Tamil translitration.

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
    const response = await ai.models.generateContent({
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
    return result;
  } catch (error) {
    console.error("Tamil Reconstruction Failed:", error);
    // Fallback to basic join if AI fails
    return {
      modernTamil: candidates.map(c => c.predictions[0].char).join(''),
      historicalTamil: candidates.map(c => c.predictions[0].char).join(''),
      phoneticTransliteration: candidates.map(c => c.predictions[0].char).join(''),
      confidence: 0.5,
      grammarCorrectionNote: "AI processing failed, using raw glyph mapping.",
      recoveredWordBoundaries: []
    };
  }
}
