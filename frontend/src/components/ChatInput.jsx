import { useRef, useEffect } from 'react'
import { Send, CornerDownLeft } from 'lucide-react'

const SUGGESTIONS = [
  'Best gyms in Austin, TX',
  'Starbucks reviews in Seattle',
  'Top 5 hotels in Miami',
  'McDonald\'s in New York',
]

export default function ChatInput({ onSend, isLoading, disabled, placeholder, showSuggestions }) {
  const textareaRef = useRef(null)

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [isLoading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    const val = textareaRef.current?.value?.trim()
    if (!val || isLoading || disabled) return
    onSend(val)
    textareaRef.current.value = ''
    textareaRef.current.style.height = 'auto'
  }

  const handleInput = () => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
  }

  const handleSuggestion = (s) => {
    if (textareaRef.current) {
      textareaRef.current.value = s
      textareaRef.current.focus()
      handleInput()
    }
  }

  return (
    <div>
      {/* Suggestion chips */}
      {showSuggestions && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          marginBottom: 10,
          paddingBottom: 10,
          borderBottom: '1px solid var(--border)',
        }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.04em' }}>
            TRY:
          </span>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => handleSuggestion(s)}
              style={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                borderRadius: 99,
                padding: '4px 12px',
                fontSize: 12,
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.15s',
                fontFamily: 'DM Sans, sans-serif',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'var(--border-active)'
                e.currentTarget.style.color = 'var(--text-primary)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border)'
                e.currentTarget.style.color = 'var(--text-secondary)'
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input row */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={1}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            disabled={isLoading || disabled}
            placeholder={placeholder || 'Ask about any business — e.g. "Best coffee shops in Chicago"'}
            style={{ paddingRight: 44 }}
          />
          <div style={{
            position: 'absolute',
            right: 12,
            bottom: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            pointerEvents: 'none',
          }}>
            <CornerDownLeft size={12} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
          </div>
        </div>

        <button
          className="send-btn"
          onClick={handleSend}
          disabled={isLoading || disabled}
          title="Send (Enter)"
        >
          {isLoading
            ? <div style={{
                width: 16, height: 16, border: '2px solid rgba(255,255,255,0.3)',
                borderTopColor: 'white', borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
            : <Send size={16} />
          }
        </button>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
