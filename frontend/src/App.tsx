import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type ModelChoice = "Gemini" | "OpenAI" | "ZhipuAI";

type ExtractedMetrics = {
  report_summary: string;
  t_stage: string;
  n_stage: string;
  crm_status: string;
  emvi_status: string;
  tumor_deposits: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ReportListItem = {
  id: number;
  source_filename: string | null;
  report_summary: string | null;
  t_stage: string | null;
  n_stage: string | null;
  crm_status: string | null;
  emvi_status: string | null;
  tumor_deposits: string | null;
  created_at: string | null;
};

type HistoryMessage = {
  id: number;
  role: "user" | "assistant";
  message: string;
  created_at: string | null;
};

type VisualSlide = {
  src: string;
  title: string;
  caption: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const MOCK_USER_ID = 1;
const visualSlides: VisualSlide[] = [
  {
    src: "/images/T.png",
    title: "T-Stage Diagram",
    caption: "T-Stage: how far the tumour has grown through the bowel wall",
  },
  {
    src: "/images/CRM.png",
    title: "CRM Diagram",
    caption: "CRM: the 'safety border' where the surgeon cuts around the tumour",
  },
  {
    src: "/images/EMVI.png",
    title: "EMVI Diagram",
    caption: "EMVI (Extramural Venous Invasion): tumour spread along tiny blood vessels",
  },
  {
    src: "/images/TD.png",
    title: "Tumour Deposit Diagram",
    caption: "Tumour Deposits: small clusters of cancer cells in the fat around the bowel (Source: adapted from published medical illustrations)",
  },
];

const emptyMetrics: ExtractedMetrics = {
  report_summary: "",
  t_stage: "",
  n_stage: "",
  crm_status: "",
  emvi_status: "",
  tumor_deposits: "",
};

const palette = {
  pageBg: "#ffffff",
  subBg: "#f8faf9",
  cardSurface: "#f3f5f4",
  border: "#d7ddd9",
  borderStrong: "#c7cfca",
  textPrimary: "#111418",
  textSecondary: "#4c5651",
  textTertiary: "#6f7a74",
  emeraldSignal: "#00d992",
  voltMint: "#2fd6a1",
  darkButton: "#0f1412",
};

export default function App() {
  const [modelChoice, setModelChoice] = useState<ModelChoice>("Gemini");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(true);
  const [isVisualOpen, setIsVisualOpen] = useState(false);
  const [currentVisualIndex, setCurrentVisualIndex] = useState(0);
  const [reportId, setReportId] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<ExtractedMetrics>(emptyMetrics);
  const [pastReports, setPastReports] = useState<ReportListItem[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hello. I am your MRI Companion AI assistant. I am here to help you understand your MRI report after you have spoken with your consultant, doctor or nurse. Upload your report below, then ask me what each result means in simple words.",
    },
  ]);

  const canUpload = useMemo(() => Boolean(selectedFile) && !isUploading, [selectedFile, isUploading]);
  const canSendMessage = useMemo(
    () => chatInput.trim().length > 0 && !isChatLoading && reportId !== null,
    [chatInput, isChatLoading, reportId],
  );
  const canLoadHistory = useMemo(() => !isLoadingHistory, [isLoadingHistory]);
  const activeSlide = visualSlides[currentVisualIndex];
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void loadReports();
  }, []);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, isChatLoading]);

  const loadReports = async () => {
    setIsLoadingHistory(true);
    try {
      const response = await fetch(`${API_BASE}/reports/${MOCK_USER_ID}`);
      if (!response.ok) {
        throw new Error(`History load failed (${response.status})`);
      }
      const data = (await response.json()) as ReportListItem[];
      setPastReports(data);
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown history error";
      setMessages((prev) => [...prev, { role: "assistant", content: `History error: ${text}` }]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleSelectReport = async (report: ReportListItem) => {
    setReportId(report.id);
    setMetrics({
      report_summary: report.report_summary ?? "No summary available.",
      t_stage: report.t_stage ?? "Not specified",
      n_stage: report.n_stage ?? "Not specified",
      crm_status: report.crm_status ?? "Not mentioned",
      emvi_status: report.emvi_status ?? "Not mentioned",
      tumor_deposits: report.tumor_deposits ?? "Not mentioned",
    });

    setIsChatLoading(true);
    try {
      const response = await fetch(`${API_BASE}/reports/${report.id}/chat-history`);
      if (!response.ok) {
        throw new Error(`Chat history load failed (${response.status})`);
      }
      const data = (await response.json()) as HistoryMessage[];
      if (data.length === 0) {
        setMessages([
          {
            role: "assistant",
            content: "This report has no previous chat. Ask a question to start a new conversation.",
          },
        ]);
        return;
      }
      setMessages(data.map((row) => ({ role: row.role, content: row.message })));
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown chat history error";
      setMessages((prev) => [...prev, { role: "assistant", content: `History error: ${text}` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("language", "English");

    setIsUploading(true);
    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Upload failed (${response.status})`);
      }
      const data = await response.json();
      setReportId(data.report_id);
      setMetrics({
        report_summary: data.summary ?? data.extracted_json?.report_summary ?? "No summary available.",
        t_stage: data.extracted_json?.t_stage ?? "Not specified",
        n_stage: data.extracted_json?.n_stage ?? "Not specified",
        crm_status: data.extracted_json?.crm_status ?? "Not mentioned",
        emvi_status: data.extracted_json?.emvi_status ?? "Not mentioned",
        tumor_deposits: data.extracted_json?.tumor_deposits ?? "Not mentioned",
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Report uploaded successfully. I am ready to answer questions about report #${data.report_id}.`,
        },
      ]);
      await loadReports();
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown upload error";
      setMessages((prev) => [...prev, { role: "assistant", content: `Upload error: ${text}` }]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    const userMessage = chatInput.trim();
    if (!userMessage || reportId === null) return;

    setChatInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsChatLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/${reportId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          model_choice: modelChoice,
        }),
      });
      if (!response.ok) {
        throw new Error(`Chat request failed (${response.status})`);
      }
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer ?? "No response." }]);
    } catch (error) {
      const text = error instanceof Error ? error.message : "Unknown chat error";
      setMessages((prev) => [...prev, { role: "assistant", content: `Chat error: ${text}` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: palette.pageBg,
        color: palette.textPrimary,
        fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
      }}
    >
      <nav
        style={{
          borderBottom: `1px solid ${palette.border}`,
          padding: "14px 20px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          backgroundColor: palette.subBg,
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700 }}>
          MRI <span style={{ color: palette.emeraldSignal }}>Companion</span>
        </h1>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: "0.9rem", color: palette.textTertiary }}>Model</span>
          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value as ModelChoice)}
            style={{
              backgroundColor: "#ffffff",
              color: palette.textPrimary,
              border: `1px solid ${palette.borderStrong}`,
              borderRadius: 8,
              padding: "8px 10px",
            }}
          >
            <option value="Gemini">Gemini</option>
            <option value="OpenAI">OpenAI</option>
            <option value="ZhipuAI">ZhipuAI</option>
          </select>
        </label>
      </nav>

      <main
        style={{
          padding: 20,
          display: "grid",
          gridTemplateColumns: isHistoryOpen ? "280px 1fr 1fr" : "84px 1fr 1fr",
          gap: 16,
          minHeight: "calc(100vh - 68px)",
        }}
      >
        <aside
          style={{
            backgroundColor: palette.cardSurface,
            border: `1px solid ${palette.border}`,
            borderRadius: 12,
            padding: 12,
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
            {isHistoryOpen && (
              <div style={{ color: palette.emeraldSignal, fontWeight: 700, fontSize: "0.95rem" }}>Past MRI Reports</div>
            )}
            <button
              type="button"
              onClick={() => setIsHistoryOpen((prev) => !prev)}
              style={{
                border: `1px solid ${palette.borderStrong}`,
                backgroundColor: "#ffffff",
                color: palette.textPrimary,
                borderRadius: 8,
                padding: "6px 10px",
                cursor: "pointer",
              }}
            >
              {isHistoryOpen ? "Collapse" : "History"}
            </button>
          </div>
          {isHistoryOpen && (
            <>
              <button
                type="button"
                onClick={loadReports}
                disabled={!canLoadHistory}
                style={{
                  marginBottom: 10,
                  border: "none",
                  borderRadius: 8,
                  padding: "8px 10px",
                  backgroundColor: canLoadHistory ? palette.darkButton : "#9aa8a2",
                  color: canLoadHistory ? palette.voltMint : "#f8faf9",
                  fontWeight: 700,
                  cursor: canLoadHistory ? "pointer" : "not-allowed",
                }}
              >
                {isLoadingHistory ? "Loading..." : "Refresh History"}
              </button>
              <div style={{ overflowY: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
                {pastReports.map((report) => (
                  <button
                    key={report.id}
                    type="button"
                    onClick={() => {
                      void handleSelectReport(report);
                    }}
                    style={{
                      textAlign: "left",
                      border: report.id === reportId ? `2px solid ${palette.emeraldSignal}` : `1px solid ${palette.borderStrong}`,
                      backgroundColor: "#ffffff",
                      color: palette.textPrimary,
                      borderRadius: 10,
                      padding: 10,
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      {report.source_filename ?? `Report #${report.id}`}
                    </div>
                    <div style={{ fontSize: "0.78rem", opacity: 0.72 }}>
                      {report.created_at ? new Date(report.created_at).toLocaleString() : `Report ID ${report.id}`}
                    </div>
                  </button>
                ))}
                {pastReports.length === 0 && (
                  <div style={{ color: palette.textTertiary, fontSize: "0.85rem" }}>
                    No past reports yet. Upload your first MRI report.
                  </div>
                )}
              </div>
            </>
          )}
        </aside>

        <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div
            style={{
              backgroundColor: palette.cardSurface,
              border: `1px solid ${palette.border}`,
              borderRadius: 12,
              padding: 16,
            }}
          >
            <h2 style={{ margin: "0 0 10px 0", fontSize: "1rem", color: palette.emeraldSignal }}>
              Upload MRI Report (PDF)
            </h2>
            <input
              id="mri-pdf-upload"
              type="file"
              accept="application/pdf"
              onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
              style={{ display: "none" }}
            />
            <label
              htmlFor="mri-pdf-upload"
              style={{
                display: "inline-block",
                marginBottom: 8,
                border: "1px solid #d1d5db",
                borderRadius: 8,
                padding: "8px 12px",
                backgroundColor: "#ffffff",
                color: palette.textPrimary,
                cursor: "pointer",
                fontSize: "0.9rem",
              }}
            >
              Choose PDF File
            </label>
            <div style={{ marginBottom: 12, fontSize: "0.86rem", color: palette.textTertiary }}>
              {selectedFile ? selectedFile.name : "No file selected"}
            </div>
            <button
              type="button"
              onClick={handleUpload}
              disabled={!canUpload}
              style={{
                width: "100%",
                border: "none",
                borderRadius: 8,
                padding: "10px 12px",
                fontWeight: 600,
                backgroundColor: canUpload ? palette.darkButton : "#9aa8a2",
                color: canUpload ? palette.voltMint : "#f8faf9",
                cursor: canUpload ? "pointer" : "not-allowed",
              }}
            >
              {isUploading ? "Extracting..." : "Start Extraction"}
            </button>
          </div>

          <div
            style={{
              backgroundColor: palette.cardSurface,
              border: `1px solid ${palette.border}`,
              borderRadius: 12,
              padding: 16,
            }}
          >
            <h2 style={{ margin: "0 0 12px 0", fontSize: "1rem", color: palette.emeraldSignal }}>
              Your MRI Results
            </h2>
            {reportId === null ? (
              <div style={{ color: palette.textTertiary, fontSize: "0.9rem" }}>
                No extracted metrics yet. Upload an MRI PDF to view results.
              </div>
            ) : (
              <>
                <Metric label="Summary" value={metrics.report_summary} />
                <Metric label="T-Stage" value={metrics.t_stage} />
                <Metric label="N-Stage" value={metrics.n_stage} />
                <Metric label="Tumor Deposits" value={metrics.tumor_deposits} />
                <Metric label="CRM" value={metrics.crm_status} />
                <Metric label="EMVI" value={metrics.emvi_status} />
              </>
            )}
          </div>

          <div
            style={{
              backgroundColor: "#ffffff",
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              boxShadow: "0 1px 2px rgba(16, 24, 40, 0.06)",
              overflow: "hidden",
            }}
          >
            <button
              type="button"
              onClick={() => setIsVisualOpen((prev) => !prev)}
              style={{
                width: "100%",
                textAlign: "left",
                border: "none",
                backgroundColor: "transparent",
                padding: "12px 14px",
                cursor: "pointer",
                color: palette.textPrimary,
                fontWeight: 600,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#f9fafb";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
              }}
            >
              {isVisualOpen ? "Hide Visual Explanations" : "View Visual Explanations"}
            </button>
            {isVisualOpen && (
              <div style={{ borderTop: "1px solid #e5e7eb", padding: 14 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, color: palette.textSecondary }}>{activeSlide.title}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <button
                      type="button"
                      onClick={() =>
                        setCurrentVisualIndex((prev) => (prev === 0 ? visualSlides.length - 1 : prev - 1))
                      }
                      style={{
                        border: "1px solid #d1d5db",
                        borderRadius: 8,
                        backgroundColor: "#ffffff",
                        color: palette.textPrimary,
                        padding: "6px 10px",
                        cursor: "pointer",
                        fontSize: "0.82rem",
                      }}
                    >
                      Previous
                    </button>
                    <span style={{ fontSize: "0.82rem", color: "#6b7280", minWidth: 52, textAlign: "center" }}>
                      {currentVisualIndex + 1}/{visualSlides.length}
                    </span>
                    <button
                      type="button"
                      onClick={() => setCurrentVisualIndex((prev) => (prev + 1) % visualSlides.length)}
                      style={{
                        border: "1px solid #d1d5db",
                        borderRadius: 8,
                        backgroundColor: "#ffffff",
                        color: palette.textPrimary,
                        padding: "6px 10px",
                        cursor: "pointer",
                        fontSize: "0.82rem",
                      }}
                    >
                      Next
                    </button>
                  </div>
                </div>
                <img
                  src={activeSlide.src}
                  alt={activeSlide.caption}
                  style={{
                    width: "100%",
                    objectFit: "contain",
                    borderRadius: 10,
                    border: "1px solid #e5e7eb",
                    backgroundColor: "#ffffff",
                  }}
                />
                <div style={{ marginTop: 8, fontSize: "0.88rem", color: "#6b7280" }}>
                  {activeSlide.caption}
                </div>
              </div>
            )}
          </div>
        </section>

        <section
          style={{
            backgroundColor: palette.cardSurface,
            border: `1px solid ${palette.border}`,
            borderRadius: 12,
            display: "flex",
            flexDirection: "column",
            height: "calc(100vh - 108px)",
          }}
        >
          <div
            style={{
              padding: "14px 16px",
              borderBottom: `1px solid ${palette.border}`,
              color: palette.emeraldSignal,
              fontWeight: 600,
            }}
          >
            Nurse Chatbot
          </div>

          <div
            ref={chatScrollRef}
            style={{
              flex: 1,
              overflowY: "auto",
              padding: 16,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              minHeight: 0,
            }}
          >
            {messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                style={{
                  maxWidth: "85%",
                  alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
                  backgroundColor: msg.role === "user" ? "#e6faf2" : "#ffffff",
                  border: msg.role === "user" ? `1px solid ${palette.voltMint}` : `1px solid ${palette.borderStrong}`,
                  color: palette.textPrimary,
                  borderRadius: 10,
                  padding: "9px 11px",
                  fontSize: "0.95rem",
                  lineHeight: 1.4,
                }}
              >
                {msg.role === "assistant" ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => (
                        <p style={{ margin: "0 0 10px 0", lineHeight: 1.55 }}>{children}</p>
                      ),
                      ul: ({ children }) => (
                        <ul style={{ margin: "0 0 10px 18px", padding: 0, listStyleType: "disc" }}>
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol style={{ margin: "0 0 10px 18px", padding: 0, listStyleType: "decimal" }}>
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => <li style={{ marginBottom: 4 }}>{children}</li>,
                      strong: ({ children }) => <strong style={{ fontWeight: 700 }}>{children}</strong>,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            ))}
            {isChatLoading && <div style={{ color: palette.textTertiary }}>Thinking...</div>}
          </div>

          <form
            onSubmit={handleSendMessage}
            style={{
              display: "flex",
              gap: 8,
              padding: 12,
              borderTop: `1px solid ${palette.border}`,
            }}
          >
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder={reportId ? "Ask about your MRI in simple words..." : "Upload a report first to start chat..."}
              disabled={reportId === null}
              style={{
                flex: 1,
                backgroundColor: "#ffffff",
                color: palette.textPrimary,
                border: `1px solid ${palette.borderStrong}`,
                borderRadius: 8,
                padding: "10px 12px",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={!canSendMessage}
              style={{
                border: "none",
                borderRadius: 8,
                padding: "10px 14px",
                backgroundColor: canSendMessage ? palette.darkButton : "#9aa8a2",
                color: canSendMessage ? palette.voltMint : "#f8faf9",
                fontWeight: 700,
                cursor: canSendMessage ? "pointer" : "not-allowed",
              }}
            >
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        border: `1px solid ${palette.border}`,
        borderRadius: 10,
        padding: 10,
        marginBottom: 8,
        backgroundColor: "#ffffff",
      }}
    >
      <div style={{ fontSize: "0.8rem", color: palette.textTertiary, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: "0.95rem", color: palette.textPrimary }}>
        {value}
      </div>
    </div>
  );
}
