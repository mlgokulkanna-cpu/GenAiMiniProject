import { Search, Brain, ShieldCheck, Zap } from 'lucide-react'

const FEATURES = [
  {
    icon: Search,
    color: '#22d3ee',
    bg: 'rgba(34,211,238,0.1)',
    title: 'Real-Time Data',
    desc: 'SerpAPI fetches live Google Maps reviews and business info',
  },
  {
    icon: Brain,
    color: '#a5b4fc',
    bg: 'rgba(165,180,252,0.1)',
    title: 'Groq Analysis',
    desc: 'llama-3.1-8b-instant runs sentiment analysis in <200ms',
  },
  {
    icon: ShieldCheck,
    color: '#10b981',
    bg: 'rgba(16,185,129,0.1)',
    title: 'Zero Vagueness',
    desc: 'Pydantic structured outputs — every verdict is data-backed',
  },
  {
    icon: Zap,
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.1)',
    title: 'Missing Info Loop',
    desc: 'LangGraph pauses to ask for location or category when missing',
  },
]

const EXAMPLES = [
  { emoji: '🏋️', text: 'Best gyms in Austin, TX' },
  { emoji: '☕', text: 'Starbucks in Seattle' },
  { emoji: '🏨', text: 'Top 5 hotels in Miami Beach' },
  { emoji: '🍕', text: 'Best pizza places in Chicago' },
]

export default function WelcomeScreen({ onExample }) {
  return (
    <div style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '32px 24px',
      gap: 32,
    }}>
      {/* Hero */}
      <div style={{ textAlign: 'center', maxWidth: 520 }}>
        <div style={{
          width: 64,
          height: 64,
          borderRadius: 18,
          background: 'linear-gradient(135deg, #6366f1, #22d3ee)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 20px',
          fontSize: 28,
          boxShadow: '0 0 40px rgba(99,102,241,0.3)',
        }}>
          🤖
        </div>
        <h2 className="animate-fade-slide-up" style={{
          fontFamily: 'Syne, sans-serif',
          fontWeight: 800,
          fontSize: 28,
          letterSpacing: '-0.02em',
          lineHeight: 1.2,
          marginBottom: 10,
        }}>
          <span className="gradient-text">Multi-Agent</span> Business Intelligence
        </h2>
        <p className="animate-fade-slide-up delay-100" style={{
          color: 'var(--text-secondary)',
          fontSize: 14,
          lineHeight: 1.65,
        }}>
          Powered by LangGraph orchestration, Groq LLM analysis, and real-time SerpAPI data.
          Ask about any business — get structured, data-backed insights instantly.
        </p>
      </div>

      {/* Feature grid */}
      <div className="animate-fade-slide-up delay-200" style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: 10,
        width: '100%',
        maxWidth: 520,
      }}>
        {FEATURES.map((f) => {
          const Icon = f.icon
          return (
            <div key={f.title} style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 12,
              padding: '12px 14px',
              display: 'flex',
              gap: 10,
              alignItems: 'flex-start',
            }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: f.bg,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}>
                <Icon size={15} color={f.color} />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', marginBottom: 2 }}>
                  {f.title}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {f.desc}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Example queries */}
      <div className="animate-fade-slide-up delay-300" style={{ width: '100%', maxWidth: 520 }}>
        <p style={{
          fontSize: 11,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--text-muted)',
          letterSpacing: '0.06em',
          marginBottom: 10,
          textAlign: 'center',
        }}>
          EXAMPLE QUERIES
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.text}
              onClick={() => onExample(ex.text)}
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 10,
                padding: '10px 12px',
                cursor: 'pointer',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                transition: 'all 0.15s',
                fontFamily: 'DM Sans, sans-serif',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'var(--border-active)'
                e.currentTarget.style.background = 'var(--bg-elevated)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border)'
                e.currentTarget.style.background = 'var(--bg-card)'
              }}
            >
              <span style={{ fontSize: 18 }}>{ex.emoji}</span>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.3 }}>{ex.text}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
