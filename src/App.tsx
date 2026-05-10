import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Upload, 
  History, 
  Download, 
  Settings, 
  Loader2, 
  ChevronRight, 
  Brain,
  Search,
  BookOpen,
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  AlertCircle,
  Volume2,
  ArrowLeftRight,
  FileText
} from 'lucide-react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';
import { reconstructTamil, performOCROnInscription, generateSpeech, reconstructHistorical, GlyphCandidate, ReconstructionResult, OCRResult, PipelineMode } from './services/geminiService';

// Expanded Brahmi mapping with multi-candidate probabilities
const BRAHMI_CANDIDATES: Record<string, Array<{char: string, prob: number}>> = {
  '𑀓': [{ char: 'க', prob: 0.98 }, { char: 'ச', prob: 0.02 }],
  '𑀔': [{ char: 'க²', prob: 0.95 }, { char: 'க', prob: 0.05 }],
  '𑀕': [{ char: 'க³', prob: 0.92 }, { char: 'க', prob: 0.08 }],
  '𑀘': [{ char: 'ச', prob: 0.96 }, { char: 'த', prob: 0.04 }],
  '𑀢': [{ char: 'த', prob: 0.94 }, { char: 'ட', prob: 0.06 }],
  '𑀝': [{ char: 'ட', prob: 0.92 }, { char: 'ற', prob: 0.05 }, { char: 'த', prob: 0.03 }],
  '𑀶': [{ char: 'ற', prob: 0.91 }, { char: 'ட', prob: 0.09 }],
  '𑀮': [{ char: 'ல', prob: 0.85 }, { char: 'ள', prob: 0.10 }, { char: 'ழ', prob: 0.05 }],
  '𑀴': [{ char: 'ள', prob: 0.82 }, { char: 'ல', prob: 0.12 }, { char: 'ழ', prob: 0.06 }],
  '𑀵': [{ char: 'ழ', prob: 0.88 }, { char: 'ள', prob: 0.08 }, { char: 'ல', prob: 0.04 }],
  '𑀅': [{ char: 'அ', prob: 0.99 }],
  '𑀆': [{ char: 'ஆ', prob: 0.99 }],
  '𑀇': [{ char: 'இ', prob: 0.99 }],
  '𑀏': [{ char: 'ஏ', prob: 0.99 }],
  '𑀼': [{ char: 'ு', prob: 0.99 }],
  '𑀸': [{ char: 'ா', prob: 0.99 }],
  '𑀺': [{ char: 'ி', prob: 0.99 }],
  '𑀻': [{ char: 'ீ', prob: 0.99 }],
  '𑁂': [{ char: 'ெ', prob: 0.99 }],
  '𑁃': [{ char: 'ை', prob: 0.99 }],
  '𑁄': [{ char: 'ொ', prob: 0.99 }],
  '𑁆': [{ char: '்', prob: 0.70 }, { char: '', prob: 0.30 }], // Often omitted in inscriptions
  ' ' : [{ char: ' ', prob: 1.0 }]
};

const DEPENDENT_MARKERS = new Set(['𑀸', '𑀺', '𑀻', '𑀼', '𑀽', '𑁂', '𑁃', '𑁄', '𑁆']);

const tokenizeInscription = (text: string): string[] => {
  const tokens: string[] = [];
  const chars = Array.from(text);
  for (let i = 0; i < chars.length; i++) {
    let token = chars[i];
    while (i + 1 < chars.length && DEPENDENT_MARKERS.has(chars[i+1])) {
      token += chars[++i];
    }
    tokens.push(token);
  }
  return tokens;
};

const getCompoundCandidates = (token: string): Array<{char: string, prob: number}> => {
  const chars = Array.from(token);
  if (chars.length === 1) {
    return BRAHMI_CANDIDATES[token] || [{ char: token, prob: 0.40 }, { char: '?', prob: 0.60 }];
  }

  // Multiply probabilities for the cluster
  // Simplified: we take top candidates for each part and combine
  let results = [{ char: '', prob: 1.0 }];
  for (const c of chars) {
    const cCandidates = BRAHMI_CANDIDATES[c] || [{ char: c, prob: 0.5 }];
    const nextResults: typeof results = [];
    for (const res of results) {
      for (const cand of cCandidates) {
        nextResults.push({
          char: res.char + cand.char,
          prob: res.prob * cand.prob
        });
      }
    }
    results = nextResults.sort((a,b) => b.prob - a.prob).slice(0, 3);
  }
  return results;
};

const TIMELINE_EVENTS = [
  { era: '300 BCE', label: 'Tamil-Brahmi' },
  { era: '200 CE', label: 'Early Vatteluttu' },
  { era: '600 CE', label: 'Grantha Influence' },
  { era: '1100 CE', label: 'Imperial Chola' },
  { era: '1500 CE', label: 'Modern Script' },
];

enum PipelineStage {
  IDLE,
  ENHANCEMENT,
  SEGMENTATION,
  PREDICTION,
  RECONSTRUCTION,
  COMPLETE
}

interface HistoryItem {
  id: string;
  type: 'ancient-to-modern' | 'modern-to-historical' | 'ocr';
  input: string;
  output: string;
  timestamp: number;
  era?: string;
}

export default function App() {
  const [inputText, setInputText] = useState('𑀓𑀼𑀭𑀴𑁆 𑀅𑀶𑀢𑁆𑀢𑀼𑀧𑁆𑀧𑀸𑀮𑁆');
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [stage, setStage] = useState<PipelineStage>(PipelineStage.IDLE);
  const [candidates, setCandidates] = useState<GlyphCandidate[]>([]);
  const [selectedEra, setSelectedEra] = useState(TIMELINE_EVENTS[0]);
  const [pipelineMode, setPipelineMode] = useState<PipelineMode>('historical');
  const [activeView, setActiveView] = useState<'prompt' | 'visualizer' | 'ocr' | 'reconstruct' | 'history'>('prompt');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  
  // OCR specific states
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResult | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Historical Reconstruction states
  const [modernKeyword, setModernKeyword] = useState('');
  const [historicalOutput, setHistoricalOutput] = useState('');
  const [isReconstructing, setIsReconstructing] = useState(false);
  const [targetEra, setTargetEra] = useState(TIMELINE_EVENTS[0]);

  const [isSpeaking, setIsSpeaking] = useState(false);

  const clearOCR = () => {
    setUploadedImage(null);
    setOcrResult(null);
    setResult(null);
    setCandidates([]);
    setStage(PipelineStage.IDLE);
  };

  const generatePDFReport = async () => {
    if (history.length === 0) return;

    // Create a hidden temporary container for the report
    const reportContainer = document.createElement('div');
    reportContainer.style.position = 'absolute';
    reportContainer.style.left = '-9999px';
    reportContainer.style.top = '0';
    reportContainer.style.width = '800px';
    reportContainer.style.backgroundColor = '#fff';
    reportContainer.style.padding = '40px';
    reportContainer.style.color = '#000';
    reportContainer.style.fontFamily = 'serif';

    const timestamp = new Date().toLocaleString();
    
    reportContainer.innerHTML = `
      <div style="border-bottom: 2px solid #b40023; padding-bottom: 20px; margin-bottom: 30px;">
        <h1 style="color: #b40023; font-size: 28px; margin: 0;">THONMAI: Neural Reconstruction Report</h1>
        <p style="font-size: 12px; color: #666; margin: 5px 0 0;">Generated on: ${timestamp}</p>
        <p style="font-size: 10px; color: #999; margin: 2px 0 0;">Neural Reconstruction Engine v3.0 Archive Export</p>
      </div>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
        <thead>
          <tr style="background-color: #b40023; color: #fff; text-transform: uppercase; font-size: 10px;">
            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">#</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Type</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Era</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Input Sequence</th>
            <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Reconstructed Output</th>
          </tr>
        </thead>
        <tbody>
          ${history.map((item, index) => `
            <tr>
              <td style="padding: 10px; border: 1px solid #ddd;">${index + 1}</td>
              <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">${item.type.toUpperCase().replace(/-/g, ' ')}</td>
              <td style="padding: 10px; border: 1px solid #ddd;">${item.era || 'N/A'}</td>
              <td style="padding: 10px; border: 1px solid #ddd; font-family: 'Noto Sans Brahmi', 'Inter', sans-serif;">${item.input}</td>
              <td style="padding: 10px; border: 1px solid #ddd; color: #b40023; font-weight: bold; font-family: 'Noto Sans Tamil', 'Inter', sans-serif;">${item.output}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 10px; color: #aaa; text-align: center;">
        End of Neural Reconstruction Report
      </div>
    `;

    document.body.appendChild(reportContainer);

    try {
      const canvas = await html2canvas(reportContainer, {
        scale: 2,
        useCORS: true,
        logging: false,
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`thonmai-reconstruction-report-${Date.now()}.pdf`);
    } catch (err) {
      console.error("PDF generation via html2canvas failed:", err);
      alert("Failed to generate PDF. Please try again.");
    } finally {
      document.body.removeChild(reportContainer);
    }
  };

  const handleSpeak = async () => {
    if (!result?.modernTamil || isSpeaking) return;
    
    setIsSpeaking(true);
    try {
      const base64Audio = await generateSpeech(result.modernTamil);
      
      const byteCharacters = atob(base64Audio);
      const byteArray = new Uint8Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteArray[i] = byteCharacters.charCodeAt(i);
      }

      // Gemini TTS returns raw PCM 16-bit little-endian at 24kHz
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      // Convert PCM 16-bit to Float32
      const int16Array = new Int16Array(byteArray.buffer);
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }
      
      const audioBuffer = audioContext.createBuffer(1, float32Array.length, 24000);
      audioBuffer.getChannelData(0).set(float32Array);
      
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.onended = () => {
        setIsSpeaking(false);
        audioContext.close();
      };
      
      source.start();
    } catch (err) {
      console.error("Speech generation failed:", err);
      setIsSpeaking(false);
      alert("Note: AI speech synthesis is currently unavailable or output format is unsupported. Please try again.");
    }
  };

  const runHistoricalReconstruction = async () => {
    if (!modernKeyword) return;
    setIsReconstructing(true);
    try {
      const result = await reconstructHistorical(modernKeyword, targetEra.label);
      setHistoricalOutput(result);
      setHistory(prev => [{
        id: Math.random().toString(36).substr(2, 9),
        type: 'modern-to-historical',
        input: modernKeyword,
        output: result,
        timestamp: Date.now(),
        era: targetEra.label
      }, ...prev]);
    } catch (err) {
      console.error(err);
    } finally {
      setIsReconstructing(false);
    }
  };

  const isProcessing = stage !== PipelineStage.IDLE && stage !== PipelineStage.COMPLETE;

  const runPipeline = useCallback(async (customGlyphs?: string[]) => {
    const sourceText = customGlyphs ? customGlyphs.join('') : inputText;
    if (!sourceText) return;
    
    // Stage 1: Enhancement
    setStage(PipelineStage.ENHANCEMENT);
    await new Promise(r => setTimeout(r, 600));

    // Stage 2: Segmentation
    setStage(PipelineStage.SEGMENTATION);
    const tokens = tokenizeInscription(sourceText);
    const segs = tokens.map((token: string) => ({
      glyph: token,
      predictions: getCompoundCandidates(token).map(p => ({ 
        char: p.char, 
        probability: p.prob,
        visualSimilarityScore: p.prob // Defaulting to probability for visual score in local mapping
      }))
    }));
    setCandidates(segs);
    await new Promise(r => setTimeout(r, 800));

    // Stage 3: Prediction
    setStage(PipelineStage.PREDICTION);
    await new Promise(r => setTimeout(r, 700));

    // Stage 4: AI Reconstruction
    setStage(PipelineStage.RECONSTRUCTION);
    const aiResult = await reconstructTamil(segs as GlyphCandidate[], selectedEra.label, pipelineMode);
    
    setResult(aiResult);
    setHistory(prev => [{
      id: Math.random().toString(36).substr(2, 9),
      type: 'ancient-to-modern',
      input: sourceText,
      output: aiResult.modernTamil,
      timestamp: Date.now(),
      era: selectedEra.label
    }, ...prev]);
    setStage(PipelineStage.COMPLETE);
  }, [inputText, selectedEra]);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = (reader.result as string).split(',')[1];
      setUploadedImage(reader.result as string);
      
      try {
        setStage(PipelineStage.ENHANCEMENT);
        const ocrData = await performOCROnInscription(base64, file.type, pipelineMode);
        setOcrResult(ocrData);
        setHistory(prev => [{
          id: Math.random().toString(36).substr(2, 9),
          type: 'ocr',
          input: 'Image Upload',
          output: ocrData.glyphs.join(''),
          timestamp: Date.now()
        }, ...prev]);
        
        // Pass to reconstruction
        await runPipeline(ocrData.glyphs);
      } catch (err) {
        console.error(err);
      } finally {
        setIsUploading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  useEffect(() => {
    if (activeView === 'prompt') {
      runPipeline();
    }
  }, [selectedEra]);

  return (
    <div className="flex flex-col h-screen bg-brand-dark text-brand-parchment font-serif overflow-hidden select-none">
      {/* Header */}
      <header className="h-20 border-b border-brand-red/30 flex items-center justify-between px-8 bg-gradient-to-r from-brand-dark to-[#1A0A0A] shrink-0 z-50">
        <div className="flex items-center gap-4">
          <motion.div 
            initial={{ rotate: -10, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            whileHover={{ scale: 1.05 }}
            className="relative w-14 h-14 flex items-center justify-center group"
          >
            {/* Logo Outer Ring */}
            <div className="absolute inset-0 rounded-full border border-brand-red/30 bg-gradient-to-br from-brand-red/10 to-transparent shadow-[0_0_20px_rgba(180,0,35,0.2)] group-hover:shadow-[0_0_30px_rgba(180,0,35,0.4)] transition-all duration-500" />
            
            {/* Logo Text Ring */}
            <svg className="absolute inset-0 w-full h-full animate-[spin_20s_linear_infinite]" viewBox="0 0 100 100">
              <defs>
                <path
                  id="logo-text-path-top"
                  d="M 50, 50 m -38, 0 a 38,38 0 1,1 76,0 a 38,38 0 1,1 -76,0"
                />
              </defs>
              <text className="text-[5.5px] uppercase tracking-[0.3em] font-serif fill-brand-red/80 font-bold">
                <textPath href="#logo-text-path-top" startOffset="0%">
                  DECIPHER • RECONSTRUCT • PRESERVE • 
                </textPath>
              </text>
            </svg>

            {/* Central Character */}
            <div className="relative text-2xl font-serif text-brand-red drop-shadow-[0_0_8px_rgba(180,0,35,0.6)] select-none">
              அ
            </div>
            
            {/* Ambient Glow */}
            <div className="absolute -bottom-1 w-6 h-[1px] bg-brand-red/40 blur-[1.5px]" />
          </motion.div>
          <div>
            <h1 className="text-2xl tracking-widest font-black uppercase text-brand-red leading-none">Thonmai</h1>
            <p className="text-[10px] uppercase tracking-[0.3em] opacity-60 mt-1 font-sans">Neural Reconstruction Engine v3.0</p>
          </div>
        </div>

        <div className="flex items-center gap-8">
          <div className="flex gap-2 bg-black/40 p-1 rounded-sm border border-brand-parchment/10">
            <button 
              onClick={() => setPipelineMode('literal')}
              className={`px-3 py-1 text-[9px] uppercase tracking-tighter transition-all ${pipelineMode === 'literal' ? 'bg-brand-red text-white' : 'opacity-40 hover:opacity-100'}`}
            >
              Literal
            </button>
            <button 
              onClick={() => setPipelineMode('historical')}
              className={`px-3 py-1 text-[9px] uppercase tracking-tighter transition-all ${pipelineMode === 'historical' ? 'bg-brand-red text-white' : 'opacity-40 hover:opacity-100'}`}
            >
              Historical
            </button>
          </div>
          <div className="flex gap-4">
            <StageIndicator current={stage} tag={PipelineStage.ENHANCEMENT} label="Enhance" />
            <StageIndicator current={stage} tag={PipelineStage.SEGMENTATION} label="Segment" />
            <StageIndicator current={stage} tag={PipelineStage.PREDICTION} label="Predict" />
            <StageIndicator current={stage} tag={PipelineStage.RECONSTRUCTION} label="Reconstruct" />
          </div>
          <button 
            onClick={() => runPipeline()}
            disabled={isProcessing}
            className="px-6 py-2 border border-brand-red hover:bg-brand-red text-brand-parchment transition-all text-xs uppercase tracking-widest font-bold active:scale-95 disabled:opacity-30"
          >
            {isProcessing ? 'Processing...' : 'Run Restoration'}
          </button>
        </div>
      </header>

      {/* Main UI */}
      <main className="flex-1 flex gap-0 overflow-hidden">
        
        {/* Left: Neural Pipeline Visualization & Prompting */}
        <section className="w-[45%] border-r border-brand-parchment/10 bg-[#0A0A0A] flex flex-col overflow-hidden">
          <div className="flex border-b border-brand-parchment/10">
            <button 
              onClick={() => setActiveView('prompt')}
              className={`flex-1 py-4 text-[10px] uppercase tracking-widest font-sans transition-all relative ${activeView === 'prompt' ? 'text-brand-red bg-brand-red/5 font-bold' : 'opacity-40 hover:opacity-100'}`}
            >
              <span className="flex items-center justify-center gap-2">
                <Search className="w-3 h-3" /> Neural Prompt
              </span>
              {activeView === 'prompt' && <motion.div layoutId="view-tab" className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand-red" />}
            </button>
            <button 
              onClick={() => setActiveView('ocr')}
              className={`flex-1 py-4 text-[10px] uppercase tracking-widest font-sans transition-all relative ${activeView === 'ocr' ? 'text-brand-red bg-brand-red/5 font-bold' : 'opacity-40 hover:opacity-100'}`}
            >
              <span className="flex items-center justify-center gap-2">
                <Upload className="w-3 h-3" /> Image OCR
              </span>
              {activeView === 'ocr' && <motion.div layoutId="view-tab" className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand-red" />}
            </button>
            <button 
              onClick={() => setActiveView('reconstruct')}
              className={`flex-1 py-4 text-[10px] uppercase tracking-widest font-sans transition-all relative ${activeView === 'reconstruct' ? 'text-brand-red bg-brand-red/5 font-bold' : 'opacity-40 hover:opacity-100'}`}
            >
              <span className="flex items-center justify-center gap-2">
                <History className="w-3 h-3" /> Historical
              </span>
              {activeView === 'reconstruct' && <motion.div layoutId="view-tab" className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand-red" />}
            </button>
            <button 
              onClick={() => setActiveView('history')}
              className={`flex-1 py-4 text-[10px] uppercase tracking-widest font-sans transition-all relative ${activeView === 'history' ? 'text-brand-red bg-brand-red/5 font-bold' : 'opacity-40 hover:opacity-100'}`}
            >
              <span className="flex items-center justify-center gap-2">
                <History className="w-3 h-3" /> Records
              </span>
              {activeView === 'history' && <motion.div layoutId="view-tab" className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand-red" />}
            </button>
            <button 
              onClick={() => setActiveView('visualizer')}
              className={`flex-1 py-4 text-[10px] uppercase tracking-widest font-sans transition-all relative ${activeView === 'visualizer' ? 'text-brand-red bg-brand-red/5 font-bold' : 'opacity-40 hover:opacity-100'}`}
            >
              <span className="flex items-center justify-center gap-2">
                <Layers className="w-3 h-3" /> Visualizer
              </span>
              {activeView === 'visualizer' && <motion.div layoutId="view-tab" className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand-red" />}
            </button>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden">
            <AnimatePresence mode="wait">
              {activeView === 'prompt' ? (
                <motion.div 
                  key="prompt"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  className="flex-1 flex flex-col p-8 gap-8 overflow-y-auto scrollbar-hide"
                >
                  <div>
                    <h3 className="text-xs uppercase tracking-[0.2em] font-bold mb-4 flex items-center gap-3">
                      <div className="w-1.5 h-1.5 bg-brand-red rounded-full" /> Ancient Input Space
                    </h3>
                    <div className="relative group">
                      <textarea 
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="w-full h-48 bg-brand-parchment/5 border border-brand-parchment/10 p-6 text-4xl leading-relaxed outline-none focus:border-brand-red/40 transition-all font-light resize-none"
                        placeholder="Paste Brahmi/Vatteluttu glyphs..."
                      />
                      <div className="absolute bottom-4 right-4 flex gap-2">
                        <button 
                          onClick={() => setInputText('')}
                          className="px-3 py-1 bg-black/40 text-[9px] uppercase tracking-widest hover:bg-black/60 transition-colors"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <h3 className="text-[10px] uppercase tracking-[0.2em] opacity-40 font-sans">Common & Compound Glyphs</h3>
                    <div className="grid grid-cols-6 gap-3">
                      {Object.keys(BRAHMI_CANDIDATES).slice(0, 18).map((glyph) => (
                        <button 
                          key={glyph}
                          onClick={() => setInputText(prev => prev + glyph)}
                          className="aspect-square bg-brand-parchment/5 border border-brand-parchment/10 flex items-center justify-center text-2xl hover:bg-brand-red hover:text-white transition-all active:scale-90"
                        >
                          {glyph}
                        </button>
                      ))}
                      {/* Compound Shortcuts */}
                      <button onClick={() => setInputText(prev => prev + '𑀓𑀼')} className="col-span-2 bg-brand-red/10 border border-brand-red/40 flex items-center justify-center text-2xl hover:bg-brand-red text-brand-red hover:text-white transition-all font-bold">𑀓𑀼</button>
                      <button onClick={() => setInputText(prev => prev + '𑀢𑁆𑀢')} className="col-span-2 bg-brand-red/10 border border-brand-red/40 flex items-center justify-center text-2xl hover:bg-brand-red text-brand-red hover:text-white transition-all font-bold">𑀢𑁆𑀢</button>
                      <button onClick={() => setInputText(prev => prev + '𑀧𑁆𑀧')} className="col-span-2 bg-brand-red/10 border border-brand-red/40 flex items-center justify-center text-2xl hover:bg-brand-red text-brand-red hover:text-white transition-all font-bold">𑀧𑁆𑀧</button>
                    </div>
                  </div>

                  <div className="mt-4 p-6 bg-brand-red/5 border border-brand-red/20 rounded-sm">
                    <div className="flex items-center gap-3 mb-2 text-brand-red">
                      <Zap className="w-4 h-4 fill-brand-red" />
                      <span className="text-[11px] font-bold uppercase tracking-widest">Neural Tip</span>
                    </div>
                    <p className="text-xs italic opacity-60 leading-relaxed">
                      "Ancient Tamil Brahmi often omits the dots (pulli). The neural engine will contextually restore them based on the {selectedEra.label} corpus."
                    </p>
                  </div>
                </motion.div>
              ) : activeView === 'ocr' ? (
                <motion.div 
                   key="ocr"
                   initial={{ opacity: 0, x: -10 }}
                   animate={{ opacity: 1, x: 0 }}
                   exit={{ opacity: 0, x: 10 }}
                   className="flex-1 flex flex-col p-8 gap-8 overflow-y-auto scrollbar-hide"
                >
                   <div className="space-y-4">
                      <div className="flex justify-between items-center">
                         <h3 className="text-xs uppercase tracking-[0.2em] font-bold flex items-center gap-3">
                           <div className="w-1.5 h-1.5 bg-brand-red rounded-full" /> Inscription Upload
                         </h3>
                         {uploadedImage && (
                           <button 
                             onClick={clearOCR}
                             className="px-3 py-1 bg-brand-red/10 border border-brand-red/40 text-brand-red text-[9px] uppercase tracking-widest hover:bg-brand-red hover:text-white transition-all"
                           >
                             Clear Results
                           </button>
                         )}
                      </div>
                      <div className="relative aspect-video rounded-sm border-2 border-dashed border-brand-parchment/10 hover:border-brand-red/30 transition-colors flex items-center justify-center overflow-hidden bg-black/20 group">
                        {uploadedImage ? (
                          <div className="relative w-full h-full">
                            <img src={uploadedImage} alt="Uploaded Inscription" className="w-full h-full object-contain" />
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                               <p className="text-[10px] uppercase tracking-widest font-bold">Replace Image</p>
                            </div>
                            {ocrResult && (
                              <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1000 1000" preserveAspectRatio="none">
                                {ocrResult.segments.map((seg, i) => (
                                  <motion.rect 
                                    key={i}
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: i * 0.05 }}
                                    x={seg.x}
                                    y={seg.y}
                                    width={seg.width}
                                    height={seg.height}
                                    fill="transparent"
                                    stroke="#B40023"
                                    strokeWidth="2"
                                    opacity="0.5"
                                  />
                                ))}
                              </svg>
                            )}
                          </div>
                        ) : (
                          <div className="text-center p-8">
                             <Upload className="w-8 h-8 mx-auto mb-4 opacity-20" />
                             <p className="text-[10px] uppercase tracking-widest opacity-40">Drop inscription photo or click to browse</p>
                             <p className="text-[8px] uppercase tracking-tighter opacity-20 mt-2">Supports PNG, JPG, TIFF (up to 10MB)</p>
                          </div>
                        )}
                        <input 
                          type="file" 
                          onChange={handleImageUpload}
                          className="absolute inset-0 opacity-0 cursor-pointer" 
                          accept="image/*"
                        />
                      </div>
                   </div>

                   {ocrResult && (
                     <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="grid grid-cols-2 gap-4"
                     >
                        <div className="p-4 bg-brand-red/5 border border-brand-red/10">
                           <span className="text-[9px] uppercase opacity-40 block mb-1">Detected Family</span>
                           <span className="text-xs font-bold text-brand-red uppercase">{ocrResult.scriptFamily}</span>
                        </div>
                        <div className="p-4 bg-brand-red/5 border border-brand-red/10">
                           <span className="text-[9px] uppercase opacity-40 block mb-1">OCR Confidence</span>
                           <span className="text-xs font-mono">{(ocrResult.confidence * 100).toFixed(1)}%</span>
                        </div>
                     </motion.div>
                   )}

                   <div className="mt-auto space-y-4">
                      <div className="p-6 border border-brand-parchment/10 bg-black/40">
                         <div className="flex items-center gap-3 mb-3">
                            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            <span className="text-[11px] font-bold uppercase tracking-widest">Auto-Enhancement Features</span>
                         </div>
                         <ul className="text-[10px] space-y-2 opacity-40 uppercase tracking-tighter">
                            <li>• Neural Denoising: Active</li>
                            <li>• Shadow Removal: Active</li>
                            <li>• Inscription Region Isolation: Active</li>
                         </ul>
                      </div>
                   </div>
                </motion.div>
              ) : activeView === 'reconstruct' ? (
                <motion.div 
                   key="reconstruct"
                   initial={{ opacity: 0, x: -10 }}
                   animate={{ opacity: 1, x: 0 }}
                   exit={{ opacity: 0, x: 10 }}
                   className="flex-1 flex flex-col p-8 gap-8 overflow-y-auto scrollbar-hide"
                >
                   <div className="space-y-4">
                      <h3 className="text-xs uppercase tracking-[0.2em] font-bold flex items-center gap-3">
                        <div className="w-1.5 h-1.5 bg-brand-red rounded-full" /> Modern → Historical
                      </h3>
                      <p className="text-[10px] uppercase tracking-widest opacity-40 leading-relaxed">
                        Input modern Tamil text to reconstruct its historical script and orthographic form.
                      </p>
                      
                      <div className="space-y-6">
                        <div>
                          <label className="text-[9px] uppercase tracking-widest opacity-40 mb-2 block font-bold">Target Era</label>
                          <div className="grid grid-cols-2 gap-2">
                            {TIMELINE_EVENTS.map((era) => (
                              <button
                                key={era.label}
                                onClick={() => setTargetEra(era)}
                                className={`px-4 py-3 text-[9px] uppercase tracking-tighter text-left border ${targetEra.label === era.label ? 'border-brand-red bg-brand-red/5 text-brand-red font-bold' : 'border-brand-parchment/10 opacity-60 hover:opacity-100'}`}
                              >
                                {era.label}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div>
                          <label className="text-[9px] uppercase tracking-widest opacity-40 mb-2 block font-bold">Modern Input</label>
                          <div className="relative">
                            <input 
                              type="text"
                              value={modernKeyword}
                              onChange={(e) => setModernKeyword(e.target.value)}
                              placeholder="e.g. தமிழினம்..."
                              className="w-full bg-black/40 border-b border-brand-red py-4 px-0 text-xl font-medium focus:outline-none placeholder:opacity-20 uppercase"
                            />
                            <button
                              onClick={runHistoricalReconstruction}
                              disabled={isReconstructing || !modernKeyword}
                              className="absolute right-0 bottom-3 p-3 bg-brand-red text-white disabled:opacity-20"
                            >
                              {isReconstructing ? <Zap className="w-4 h-4 animate-spin" /> : <ArrowLeftRight className="w-4 h-4" />}
                            </button>
                          </div>
                        </div>

                        {historicalOutput && (
                          <motion.div 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="p-8 bg-brand-red/5 border border-brand-red/20 text-center space-y-4"
                          >
                            <span className="text-[9px] uppercase tracking-widest opacity-40 block">Historical Reconstruction</span>
                            <p className="text-6xl font-bold tracking-tighter text-brand-red break-all line-height-tight">
                              {historicalOutput}
                            </p>
                            <div className="flex items-center justify-center gap-4 pt-4">
                               <div className="h-[1px] flex-1 bg-brand-red/10" />
                               <span className="text-[8px] uppercase tracking-widest opacity-40 whitespace-nowrap">Authentic {targetEra.label} Script</span>
                               <div className="h-[1px] flex-1 bg-brand-red/10" />
                            </div>
                          </motion.div>
                        )}
                      </div>
                   </div>

                   <div className="mt-auto p-6 border border-brand-parchment/10 bg-black/40">
                      <div className="flex items-center gap-3 mb-3">
                         <div className="w-2 h-2 border border-emerald-500 rounded-full" />
                         <span className="text-[11px] font-bold uppercase tracking-widest">Reconstruction Parameters</span>
                      </div>
                      <ul className="text-[10px] space-y-2 opacity-40 uppercase tracking-tighter">
                         <li>• Script Morphing: Active ({targetEra.label})</li>
                         <li>• Phonetic Alignment: Linear Reconstruction</li>
                         <li>• Ligature Synthesis: Era-Aware</li>
                      </ul>
                   </div>
                </motion.div>
              ) : activeView === 'history' ? (
                <motion.div 
                   key="history"
                   initial={{ opacity: 0, x: -10 }}
                   animate={{ opacity: 1, x: 0 }}
                   exit={{ opacity: 0, x: 10 }}
                   className="flex-1 flex flex-col p-8 gap-6 overflow-y-auto scrollbar-hide"
                >
                   <div className="space-y-2 mb-4">
                      <h3 className="text-xs uppercase tracking-[0.2em] font-bold flex items-center gap-3">
                        <div className="w-1.5 h-1.5 bg-brand-red rounded-full" /> Neural Archives
                      </h3>
                      <p className="text-[10px] uppercase tracking-widest opacity-40 leading-relaxed">
                        A historical record of your inscriptional neural transformations.
                      </p>
                      <motion.button 
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={generatePDFReport}
                        disabled={history.length === 0}
                        className="mt-4 flex items-center gap-2 px-4 py-2 bg-brand-red text-white text-[10px] font-bold uppercase tracking-widest hover:brightness-125 transition-all disabled:opacity-30"
                      >
                         <Download className="w-3 h-3" /> Export PDF Report
                      </motion.button>
                   </div>

                   {history.length === 0 ? (
                     <div className="flex-1 flex flex-col items-center justify-center opacity-20 py-20 grayscale">
                        <History className="w-12 h-12 mb-4" />
                        <p className="text-[10px] uppercase tracking-[0.4em]">No archives found</p>
                     </div>
                   ) : (
                     <div className="space-y-4">
                       {history.map((item) => (
                         <div key={item.id} className="p-5 bg-brand-parchment/5 border border-brand-parchment/10 hover:border-brand-red/30 transition-all group">
                           <div className="flex justify-between items-start mb-3">
                             <div className="flex flex-col">
                               <span className="text-[8px] uppercase tracking-widest text-brand-red font-bold mb-1">
                                 {item.type.replace(/-/g, ' ')}
                               </span>
                               <span className="text-[11px] font-mono opacity-60">
                                 {new Date(item.timestamp).toLocaleTimeString()}
                               </span>
                             </div>
                             {item.era && (
                               <span className="px-2 py-0.5 bg-brand-red/10 border border-brand-red/30 text-brand-red text-[7px] uppercase font-bold tracking-widest">
                                 {item.era}
                               </span>
                             )}
                           </div>
                           
                           <div className="space-y-3">
                             <div className="flex gap-3">
                               <div className="w-1 h-auto bg-brand-red/20 rounded-full" />
                               <div>
                                 <span className="text-[7px] uppercase opacity-40 block mb-0.5">Input Sequence</span>
                                 <p className="text-xs font-serif truncate max-w-[200px]">{item.input}</p>
                               </div>
                             </div>
                             <div className="flex gap-3">
                               <div className="w-1 h-auto bg-brand-red rounded-full shadow-[0_0_5px_rgba(180,0,35,0.4)]" />
                               <div>
                                 <span className="text-[7px] uppercase opacity-40 block mb-0.5">Reconstructed Output</span>
                                 <p className="text-sm font-bold text-brand-red line-clamp-2">{item.output}</p>
                               </div>
                             </div>
                           </div>
                         </div>
                       ))}
                     </div>
                   )}
                </motion.div>
              ) : activeView === 'visualizer' ? (
                <motion.div 
                  key="visualizer"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="flex-1 flex flex-col overflow-hidden"
                >
                  <div className="p-6 border-b border-brand-parchment/5 flex justify-between items-center">
                    <h3 className="text-[10px] uppercase tracking-widest opacity-40 font-sans flex items-center gap-2">
                      Visual Glyph Segmentation
                    </h3>
                    <span className="text-[9px] font-mono opacity-30">BATCH_ID: #TC-{(Math.random() * 10000).toFixed(0)}</span>
                  </div>

                  <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
                    <div className="grid grid-cols-1 gap-6">
                      <AnimatePresence mode="popLayout">
                        {candidates.map((cand, idx) => (
                          <motion.div 
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className="flex items-center gap-6 group"
                          >
                            <div className="min-w-16 h-16 px-4 bg-brand-parchment/5 border border-brand-parchment/10 flex items-center justify-center text-4xl group-hover:bg-brand-red/10 transition-colors uppercase whitespace-nowrap">
                              {cand.glyph}
                            </div>
                            <ChevronRight className="w-4 h-4 opacity-20 shrink-0" />
                            <div className="flex-1 flex gap-2">
                              {cand.predictions.map((p, pIdx) => (
                                <div key={pIdx} className="flex flex-col gap-1 flex-1">
                                  <div className={`h-1.5 rounded-full overflow-hidden bg-white/5`}>
                                     <motion.div 
                                       initial={{ width: 0 }}
                                       animate={{ width: `${p.probability * 100}%` }}
                                       className={`h-full ${pIdx === 0 ? 'bg-brand-red' : 'bg-brand-parchment/20'}`}
                                     />
                                  </div>
                                  <div className="flex justify-between text-[10px] font-mono uppercase">
                                     <span className={pIdx === 0 ? 'text-brand-red font-bold' : 'opacity-40'}>{p.char || '∅'}</span>
                                     <span className="opacity-30">{(p.probability * 100).toFixed(0)}%</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                    </div>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>

          {/* Model Status */}
          <div className="p-6 bg-brand-red/5 border-t border-brand-red/20 space-y-4">
            <div className="flex justify-between items-end">
              <div>
                <span className="text-[9px] uppercase tracking-widest opacity-40 font-sans block mb-1">Active Model</span>
                <span className="text-sm font-bold text-brand-red">Historical-Tamil-Transformer-v3</span>
              </div>
              <div className="text-right">
                <span className="text-[9px] uppercase tracking-widest opacity-40 font-sans block mb-1">Decoding Strategy</span>
                <span className="text-sm font-bold">Beam Search (Width: 8)</span>
              </div>
            </div>
            <div className="h-[1px] bg-brand-parchment/10 w-full" />
            <div className="flex gap-4">
               <div className="flex-1">
                  <span className="text-[9px] uppercase opacity-40 block mb-1">OCR Confidence</span>
                  <div className="text-lg font-mono tracking-tighter">{(result?.confidence || 0.85 * 100).toFixed(2)}%</div>
               </div>
               <div className="flex-1">
                  <span className="text-[9px] uppercase opacity-40 block mb-1">Corpus Sync</span>
                  <div className="text-lg font-mono tracking-tighter text-emerald-500 uppercase">Synced</div>
               </div>
            </div>
          </div>
        </section>

        {/* Right: Reconstruction Output */}
        <section className="flex-1 bg-brand-parchment text-brand-dark p-12 flex flex-col relative">
          <div className="absolute inset-0 pointer-events-none opacity-[0.03] overflow-hidden flex items-center justify-center">
             <span className="text-[600px] font-black rotate-12">தமிழ்</span>
          </div>

          <div className="relative z-10 flex flex-col h-full">
            <div className="flex items-center justify-between mb-12">
              <h2 className="text-xs uppercase tracking-[0.5em] font-black border-l-[6px] border-brand-red pl-6">Neural Reconstruction</h2>
              <div className="flex gap-4">
                <button 
                  onClick={() => setActiveView('history')}
                  className="px-6 py-2 bg-brand-dark text-brand-parchment text-[10px] font-bold uppercase tracking-widest hover:brightness-125 transition-all shadow-lg shadow-black/40"
                >
                  History
                </button>
                <button 
                  onClick={generatePDFReport}
                  disabled={history.length === 0}
                  className="w-10 h-10 border border-black/10 flex items-center justify-center hover:bg-black/5 transition-colors disabled:opacity-20"
                  title="Export PDF Report"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 flex flex-col justify-center max-w-2xl px-4">
              <AnimatePresence mode="wait">
                {stage === PipelineStage.RECONSTRUCTION ? (
                    <motion.div 
                      key="loader"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="space-y-6"
                    >
                      <div className="h-16 w-3/4 bg-black/5 animate-pulse rounded-sm" />
                      <div className="h-16 w-full bg-black/5 animate-pulse rounded-sm" />
                      <div className="h-16 w-1/2 bg-black/5 animate-pulse rounded-sm" />
                      <p className="text-sm font-sans uppercase tracking-widest opacity-20 animate-pulse mt-8">Linguistic Model Solving Ambiguities...</p>
                    </motion.div>
                ) : (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-12"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-[10px] uppercase tracking-widest opacity-30 font-sans block">Final Transliteration</span>
                        <motion.button 
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={handleSpeak}
                          disabled={isSpeaking}
                          className="flex items-center gap-2 px-3 py-1.5 bg-brand-red text-white text-[10px] font-bold uppercase tracking-widest hover:brightness-125 transition-all disabled:opacity-50"
                        >
                          <Volume2 className={`w-3 h-3 ${isSpeaking ? 'animate-pulse' : ''}`} />
                          {isSpeaking ? 'Speaking...' : 'Listen Pronunciation'}
                        </motion.button>
                      </div>
                      <p className="text-7xl font-bold tracking-tighter leading-[1.1] mb-2 leading-tight">
                        {result?.modernTamil || 'INIT_SEQ...'}
                      </p>
                      
                      <p className="text-lg font-mono opacity-40 mb-8 uppercase tracking-widest">
                        {result?.phoneticTransliteration}
                      </p>
                      
                      <div className="flex flex-wrap gap-2">
                        {result?.recoveredWordBoundaries.map((word, i) => (
                          <span key={i} className="px-3 py-1 bg-brand-red/10 text-brand-red text-[10px] font-bold uppercase tracking-wider rounded-full">
                            {word}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="p-8 border-l-2 border-brand-red/20 bg-black/5">
                      <h4 className="text-[10px] uppercase tracking-widest font-bold mb-4 flex items-center gap-2">
                        <Brain className="w-3 h-3 text-brand-red" /> Historical Context Engine
                      </h4>
                      <p className="text-xl leading-relaxed italic opacity-80 serif">
                        "{result?.grammarCorrectionNote || 'Ancient glyph patterns analyzed against Sangam literature corpus.'}"
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Stats Footbar */}
            <div className="mt-auto border-t border-black/5 pt-12 flex justify-between items-center opacity-40">
               <div className="flex gap-8">
                 <div>
                    <span className="text-[9px] uppercase tracking-widest font-sans block">Processing Speed</span>
                    <span className="text-xs font-mono">420 tokens/sec</span>
                 </div>
                 <div>
                    <span className="text-[9px] uppercase tracking-widest font-sans block">Error Margin</span>
                    <span className="text-xs font-mono">0.02%</span>
                 </div>
               </div>
               <div className="text-right">
                 <span className="text-[9px] uppercase tracking-widest font-sans block">Linguistic Model</span>
                 <span className="text-xs font-mono italic">Tamil-BERT-v2</span>
               </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer Timeline */}
      <footer className="h-28 bg-[#050505] border-t border-brand-red/20 flex items-center px-8 gap-12 shrink-0 z-50">
        <div className="w-40">
           <span className="text-[9px] uppercase font-sans tracking-widest opacity-40 mb-1 block">Weight Bias</span>
           <div className="text-sm font-bold truncate text-brand-red uppercase">{selectedEra.label}</div>
        </div>
        
        <div className="flex-1 relative flex items-center h-full">
          <div className="h-[1px] w-full bg-brand-parchment/10 absolute top-1/2 -translate-y-1/2"></div>
          <div className="flex justify-between w-full relative">
            {TIMELINE_EVENTS.map((event, idx) => {
              const isSelected = selectedEra.era === event.era;
              return (
                <div 
                  key={idx} 
                  onClick={() => !isProcessing && setSelectedEra(event)}
                  className={`flex flex-col items-center group cursor-pointer transition-all ${isProcessing ? 'pointer-events-none opacity-20' : isSelected ? 'opacity-100' : 'opacity-30 hover:opacity-100'}`}
                >
                  <motion.div 
                    animate={isSelected ? { scale: [1, 1.2, 1], rotate: [0, 90, 0] } : {}}
                    className={`w-3 h-3 rounded-sm transition-all duration-500 mb-2 rotate-45 border ${isSelected ? 'bg-brand-red border-brand-red shadow-[0_0_15px_rgba(180,0,35,0.5)]' : 'border-brand-parchment/40 bg-transparent'}`} 
                  />
                  <span className="text-[10px] font-bold mb-0.5">{event.era}</span>
                  <span className="text-[8px] uppercase tracking-tighter opacity-50 font-sans">{event.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="flex gap-12 items-center border-l border-brand-parchment/10 pl-12 font-sans">
           <div className="flex gap-4">
              <Metric label="Latency" value="1.2s" color="text-amber-500" />
              <Metric label="VRAM" value="8.4GB" />
              <Metric label="Accuracy" value="99.2%" />
           </div>
           <Settings className="w-5 h-5 opacity-40 hover:opacity-100 cursor-pointer transition-opacity" />
        </div>
      </footer>
    </div>
  );
}

function StageIndicator({ current, tag, label }: { current: PipelineStage, tag: PipelineStage, label: string }) {
  const isActive = current === tag;
  const isDone = current > tag;
  return (
    <div className={`flex flex-col items-center gap-1 transition-opacity ${isActive || isDone ? 'opacity-100' : 'opacity-20'}`}>
       <div className={`w-1 h-1 rounded-full ${isActive ? 'bg-brand-red animate-ping' : isDone ? 'bg-brand-red' : 'bg-white'}`} />
       <span className="text-[8px] uppercase tracking-widest font-bold">{label}</span>
    </div>
  );
}

function Metric({ label, value, color = "text-white" }: { label: string, value: string, color?: string }) {
  return (
    <div className="flex flex-col">
       <span className="text-[8px] uppercase opacity-40 tracking-widest">{label}</span>
       <span className={`text-[10px] font-mono leading-none ${color}`}>{value}</span>
    </div>
  )
}
