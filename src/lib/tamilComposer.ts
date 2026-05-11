
/**
 * Tamil Unicode Composition Engine
 * Handles proper assembly of recognized Tamil characters into valid Unicode text.
 */

export enum CharacterType {
  BASE_CONSONANT = "base_consonant",      // க, ங, ச, ஞ, etc.
  INDEPENDENT_VOWEL = "independent_vowel", // அ, ஆ, இ, ஈ, etc.
  VOWEL_SIGN = "vowel_sign",              // ா, ி, ீ, ு, ூ, ெ, ே, ை, ொ, ோ, ௌ
  PULLI = "pulli",                        // ் (virama)
  AYTHAM = "aytham",                      // ஃ
  NUMERAL = "numeral",                    // ௦, ௧, ௨, etc.
  UNKNOWN = "unknown"
}

export interface RecognizedChar {
  character: string;
  confidence: number;
  bbox?: any;
  index?: number;
}

export interface ClassifiedChar extends RecognizedChar {
  type: CharacterType;
  originalIndex: number;
}

export class TamilUnicodeComposer {
  // Unicode ranges for Tamil
  private readonly CONSONANTS = new Set(Array.from({ length: 0x0BB9 - 0x0B95 + 1 }, (_, i) => 0x0B95 + i));
  private readonly VOWELS = new Set(Array.from({ length: 0x0B94 - 0x0B85 + 1 }, (_, i) => 0x0B85 + i));
  
  private readonly PULLI = '\u0BCD'; // ்
  private readonly AYTHAM = '\u0B83'; // ஃ
  
  private readonly VOWEL_SIGNS = new Set([
    '\u0BBE', '\u0BBF', '\u0BC0', '\u0BC1', '\u0BC2', 
    '\u0BC6', '\u0BC7', '\u0BC8', '\u0BCA', '\u0BCB', '\u0BCC'
  ]);

  // Vowel signs that display BEFORE consonant but encode AFTER
  private readonly PRE_BASE_VOWEL_SIGNS = new Set(['\u0BC6', '\u0BC7', '\u0BC8']); // ெ, ே, ை

  private debugLogs: string[] = [];

  classifyCharacter(char: string): CharacterType {
    if (!char) return CharacterType.UNKNOWN;
    const codePoint = char.charCodeAt(0);

    if (this.CONSONANTS.has(codePoint)) return CharacterType.BASE_CONSONANT;
    if (this.VOWELS.has(codePoint)) return CharacterType.INDEPENDENT_VOWEL;
    if (this.VOWEL_SIGNS.has(char)) return CharacterType.VOWEL_SIGN;
    if (char === this.PULLI) return CharacterType.PULLI;
    if (char === this.AYTHAM) return CharacterType.AYTHAM;
    if (codePoint >= 0x0BE6 && codePoint <= 0x0BEF) return CharacterType.NUMERAL;
    
    return CharacterType.UNKNOWN;
  }

  compose(recognizedChars: RecognizedChar[]) {
    this.debugLogs = [];
    const warnings: string[] = [];

    // Step 1: Classify
    const classified: ClassifiedChar[] = recognizedChars.map((charData, idx) => ({
      ...charData,
      type: this.classifyCharacter(charData.character),
      originalIndex: charData.index ?? idx
    }));

    // Step 2: Visual -> Logical Reordering
    const logicalOrder = this.reorderForLogicalEncoding(classified);

    // Step 3: Compose
    const composedUnits: string[] = [];
    let i = 0;

    while (i < logicalOrder.length) {
      const current = logicalOrder[i];
      const charType = current.type;
      const char = current.character;

      if (charType === CharacterType.BASE_CONSONANT) {
        const { unit, consumed } = this.composeConsonantCluster(logicalOrder.slice(i));
        composedUnits.push(unit);
        i += consumed;
      } else if (charType === CharacterType.INDEPENDENT_VOWEL) {
        composedUnits.push(char);
        i++;
      } else if (charType === CharacterType.VOWEL_SIGN || charType === CharacterType.PULLI) {
        // Orphaned modifiers
        warnings.push(`Orphaned ${charType} '${char}' at index ${i}`);
        const lastUnit = composedUnits[composedUnits.length - 1];
        if (lastUnit && this.isConsonant(lastUnit[0])) {
          composedUnits[composedUnits.length - 1] += char;
          this.debugLogs.push(`Attached orphaned '${char}' to previous unit`);
        } else {
          composedUnits.push(char);
        }
        i++;
      } else {
        composedUnits.push(char);
        i++;
      }
    }

    // Step 4: Join and normalize
    const composedText = composedUnits.join('').normalize('NFC');
    
    // Step 5: Validate
    const validationWarnings = this.validateTamilText(composedText);
    warnings.push(...validationWarnings);
    
    return {
      composedText,
      debugSteps: this.debugLogs,
      warnings,
      stats: {
        inputChars: recognizedChars.length,
        outputUnits: composedUnits.length,
        outputLength: composedText.length
      }
    };
  }

  private reorderForLogicalEncoding(chars: ClassifiedChar[]): ClassifiedChar[] {
    const reordered: ClassifiedChar[] = [];
    let i = 0;

    while (i < chars.length) {
      const current = chars[i];
      
      if (current.type === CharacterType.VOWEL_SIGN && this.PRE_BASE_VOWEL_SIGNS.has(current.character)) {
        if (i + 1 < chars.length && chars[i + 1].type === CharacterType.BASE_CONSONANT) {
          reordered.push(chars[i + 1]);
          reordered.push(current);
          this.debugLogs.push(`Reordered: ${current.character} + ${chars[i+1].character} -> ${chars[i+1].character} + ${current.character}`);
          i += 2;
        } else {
          reordered.push(current);
          i++;
        }
      } else {
        reordered.push(current);
        i++;
      }
    }
    return reordered;
  }

  private composeConsonantCluster(chars: ClassifiedChar[]): { unit: string, consumed: number } {
    const base = chars[0].character;
    let unit = base;
    let consumed = 1;

    if (chars.length > 1) {
      if (chars[1].type === CharacterType.VOWEL_SIGN) {
        unit += chars[1].character;
        consumed = 2;
        if (chars.length > 2 && chars[2].type === CharacterType.PULLI) {
          unit += this.PULLI;
          consumed = 3;
        }
      } else if (chars[1].type === CharacterType.PULLI) {
        unit += this.PULLI;
        consumed = 2;
      }
    }

    return { unit, consumed };
  }

  private validateTamilText(text: string): string[] {
    const warnings: string[] = [];
    let prevCharType: CharacterType | null = null;
    let prevChar: string | null = null;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const charType = this.classifyCharacter(char);

      // Rule: Vowel sign cannot follow another vowel sign
      if (prevCharType === CharacterType.VOWEL_SIGN && charType === CharacterType.VOWEL_SIGN) {
        warnings.push(`Invalid: Two consecutive vowel signs ('${prevChar}${char}') at pos ${i}`);
      }

      // Rule: Vowel sign cannot follow independent vowel
      if (prevCharType === CharacterType.INDEPENDENT_VOWEL && charType === CharacterType.VOWEL_SIGN) {
        warnings.push(`Invalid: Vowel sign '${char}' follows independent vowel '${prevChar}'`);
      }

      // Rule: Pulli cannot follow vowel sign
      if (prevCharType === CharacterType.VOWEL_SIGN && charType === CharacterType.PULLI) {
        warnings.push(`Unusual: Pulli after vowel sign at position ${i}`);
      }

      // Rule: Vowel sign or pulli cannot be at the start of text
      if (i === 0 && (charType === CharacterType.VOWEL_SIGN || charType === CharacterType.PULLI)) {
        warnings.push(`Warning: Text starts with a modifier '${char}'`);
      }

      prevCharType = charType;
      prevChar = char;
    }

    return warnings;
  }

  private isConsonant(char: string): boolean {
    return this.CONSONANTS.has(char.charCodeAt(0));
  }
}

export const assembleTamilText = (chars: RecognizedChar[]) => {
  const composer = new TamilUnicodeComposer();
  return composer.compose(chars);
};
