import { useState } from 'react'
import { MapPin, Phone, Globe, Clock, ThumbsUp, ThumbsDown, Star, ChevronDown, ChevronUp } from 'lucide-react'
import ScoreRing from './ScoreRing'

const RECOMMENDATION_STYLES = {
  HIGHLY_RECOMMENDED: { label: 'Highly Recommended', bg: 'rgba(16,185,129,0.15)', color: '#10b981', border: 'rgba(16,185,129,0.3)' },
  RECOMMENDED:        { label: 'Recommended',         bg: 'rgba(99,102,241,0.15)', color: '#a5b4fc', border: 'rgba(99,102,241,0.3)' },
  NEUTRAL:            { label: 'Neutral',              bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: 'rgba(245,158,11,0.3)' },
  AVOID:              { label: 'Avoid',                bg: 'rgba(244,63,94,0.15)',  color: '#f43f5e', border: 'rgba(244,63,94,0.3)' },
}

function SentimentBar({ breakdown }) {
  const total = (breakdown.positive || 0) + (breakdown.neutral || 0) + (breakdown.negative || 0)
  if (total === 0) return null
  const pct = (v) => Math.round((v / total) * 100)

  return (
    <div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.04em' }}>
        SENTIMENT BREAKDOWN
      </p>
      <div style={{ display: 'flex', height: 8, borderRadius: 99, overflow: 'hidden', gap: 2 }}>
        {breakdown.positive > 0 && (
          <div title={`Positive: ${pct(breakdown.positive)}%`}
            style={{ flex: breakdown.positive, background: '#10b981', borderRadius: 99, transition: 'flex 0.6s ease' }} />
        )}
        {breakdown.neutral > 0 && (
          <div title={`Neutral: ${pct(breakdown.neutral)}%`}
            style={{ flex: breakdown.neutral, background: '#f59e0b', borderRadius: 99 }} />
        )}
        {breakdown.negative > 0 && (
          <div title={`Negative: ${pct(breakdown.negative)}%`}
            style={{ flex: breakdown.negative, background: '#f43f5e', borderRadius: 99 }} />
        )}
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11, color: 'var(--text-secondary)' }}>
        <span style={{ color: '#10b981' }}>▲ {pct(breakdown.positive)}% positive</span>
        <span style={{ color: '#f59e0b' }}>● {pct(breakdown.neutral)}% neutral</span>
        <span style={{ color: '#f43f5e' }}>▼ {pct(breakdown.negative)}% negative</span>
      </div>
    </div>
  )
}

function ReviewHighlight({ highlight }) {
  const colors = { positive: '#10b981', neutral: '#f59e0b', negative: '#f43f5e' }
  const color = colors[highlight.sentiment] || '#8888aa'

  return (
    <div style={{
      background: 'var(--bg-elevated)',
      borderLeft: `3px solid ${color}`,
      borderRadius: '0 8px 8px 0',
      padding: '10px 12px',
      fontSize: 13,
      color: 'var(--text-secondary)',
      fontStyle: 'italic',
      lineHeight: 1.5,
    }}>
      "{highlight.text}"
      <div style={{ marginTop: 4, fontSize: 11, fontStyle: 'normal', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
        #{highlight.theme}
      </div>
    </div>
  )
}

export default function BusinessCard({ data, rank, isWinner }) {
  const [expanded, setExpanded] = useState(!rank || rank === 1)
  const rec = RECOMMENDATION_STYLES[data.recommendation] || RECOMMENDATION_STYLES.NEUTRAL

  return (
    <div className="glass-card animate-scale-in" style={{
      overflow: 'hidden',
      border: isWinner ? '1px solid rgba(99,102,241,0.4)' : '1px solid var(--border)',
      boxShadow: isWinner ? '0 0 32px rgba(99,102,241,0.12)' : 'none',
    }}>
      {/* Winner ribbon */}
      {isWinner && (
        <div style={{
          background: 'linear-gradient(90deg, #6366f1, #22d3ee)',
          padding: '5px 16px',
          fontSize: 11,
          fontFamily: 'JetBrains Mono, monospace',
          fontWeight: 600,
          letterSpacing: '0.08em',
          color: 'white',
        }}>
          🏆 TOP PICK
        </div>
      )}

      {/* Header */}
      <div style={{ padding: '18px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          {rank && (
            <div style={{
              fontFamily: 'Syne, sans-serif',
              fontWeight: 800,
              fontSize: 28,
              color: isWinner ? '#6366f1' : 'var(--text-muted)',
              lineHeight: 1,
              minWidth: 32,
              marginTop: 4,
            }}>
              #{rank}
            </div>
          )}

          <ScoreRing score={data.overall_score} size={rank ? 64 : 72} />

          <div style={{ flex: 1, minWidth: 0 }}>
            <h3 style={{
              fontFamily: 'Syne, sans-serif',
              fontWeight: 700,
              fontSize: 18,
              color: 'var(--text-primary)',
              marginBottom: 4,
              lineHeight: 1.2,
            }}>
              {data.business_name}
            </h3>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
              {data.rating && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, color: '#f59e0b' }}>
                  <Star size={12} fill="#f59e0b" />
                  <span style={{ fontWeight: 600 }}>{data.rating}</span>
                  {data.total_reviews && (
                    <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
                      ({data.total_reviews.toLocaleString()} reviews)
                    </span>
                  )}
                </div>
              )}
              <div style={{
                ...rec,
                padding: '2px 10px',
                borderRadius: 99,
                fontSize: 11,
                fontWeight: 600,
                border: `1px solid ${rec.border}`,
                letterSpacing: '0.02em',
              }}>
                {rec.label}
              </div>
            </div>

            {/* Quick info */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {data.address && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-muted)' }}>
                  <MapPin size={11} /> {data.address}
                </div>
              )}
              {data.phone && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-muted)' }}>
                  <Phone size={11} /> {data.phone}
                </div>
              )}
            </div>
          </div>

          {/* Expand toggle */}
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '6px 8px',
              cursor: 'pointer',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              flexShrink: 0,
            }}
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '18px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: 18,
        }}>
          {/* Verdict */}
          <div style={{
            background: 'var(--bg-elevated)',
            borderRadius: 10,
            padding: '12px 14px',
            fontSize: 13.5,
            color: 'var(--text-secondary)',
            lineHeight: 1.65,
            borderLeft: '3px solid var(--accent)',
          }}>
            {data.verdict}
          </div>

          {/* Sentiment bar */}
          {data.sentiment_breakdown && <SentimentBar breakdown={data.sentiment_breakdown} />}

          {/* Pros & Cons */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <p style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#10b981', marginBottom: 8, letterSpacing: '0.04em' }}>
                ✓ PROS
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {(data.pros || []).map((pro, i) => (
                  <div key={i} style={{
                    display: 'flex', gap: 8, fontSize: 13,
                    color: 'var(--text-secondary)', alignItems: 'flex-start'
                  }}>
                    <ThumbsUp size={12} color="#10b981" style={{ marginTop: 2, flexShrink: 0 }} />
                    {pro}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: '#f43f5e', marginBottom: 8, letterSpacing: '0.04em' }}>
                ✗ CONS
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {(data.cons || []).map((con, i) => (
                  <div key={i} style={{
                    display: 'flex', gap: 8, fontSize: 13,
                    color: 'var(--text-secondary)', alignItems: 'flex-start'
                  }}>
                    <ThumbsDown size={12} color="#f43f5e" style={{ marginTop: 2, flexShrink: 0 }} />
                    {con}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top themes */}
          {data.top_themes?.length > 0 && (
            <div>
              <p style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.04em' }}>
                KEY THEMES
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {data.top_themes.map((t, i) => (
                  <span key={i} style={{
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)',
                    borderRadius: 99,
                    padding: '3px 10px',
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                  }}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Review highlights */}
          {data.review_highlights?.length > 0 && (
            <div>
              <p style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.04em' }}>
                REVIEW HIGHLIGHTS
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {data.review_highlights.slice(0, 3).map((h, i) => (
                  <ReviewHighlight key={i} highlight={h} />
                ))}
              </div>
            </div>
          )}

          {/* Contact links */}
          <div style={{ display: 'flex', gap: 10 }}>
            {data.website && (
              <a href={data.website} target="_blank" rel="noopener noreferrer"
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontSize: 12, color: 'var(--accent-cyan)',
                  textDecoration: 'none', padding: '5px 12px',
                  background: 'rgba(34,211,238,0.08)',
                  border: '1px solid rgba(34,211,238,0.2)',
                  borderRadius: 8,
                }}>
                <Globe size={12} /> Visit Website
              </a>
            )}
            {data.hours && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                fontSize: 12, color: 'var(--text-muted)',
                padding: '5px 12px',
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                borderRadius: 8,
              }}>
                <Clock size={12} /> {data.hours}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
