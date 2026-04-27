import { useState, useCallback, useRef } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const generateSessionId = () =>
  `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`

export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [agentState, setAgentState] = useState('idle')
  const [resultData, setResultData] = useState(null)
  const [error, setError] = useState(null)
  const sessionIdRef = useRef(generateSessionId())

  const addMessage = useCallback((role, content, extra = {}) => {
    setMessages(prev => [
      ...prev,
      { id: Date.now() + Math.random(), role, content, timestamp: new Date(), ...extra }
    ])
  }, [])

  const sendMessage = useCallback(async (userText) => {
    if (!userText.trim() || isLoading) return

    setError(null)
    addMessage('user', userText)
    setIsLoading(true)
    setAgentState('triaging')

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        session_id: sessionIdRef.current,
        message: userText,
      })

      const data = response.data
      setAgentState(data.agent_state)

      if (data.data) {
        setResultData(data.data)
      }

      addMessage('assistant', data.message, {
        agentState: data.agent_state,
        requiresInput: data.requires_input,
        inputPrompt: data.input_prompt,
        data: data.data,
      })

    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Network error'
      setError(msg)
      addMessage('assistant', `❌ ${msg}`, { agentState: 'error' })
      setAgentState('error')
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, addMessage])

  const resetSession = useCallback(() => {
    sessionIdRef.current = generateSessionId()
    setMessages([])
    setAgentState('idle')
    setResultData(null)
    setError(null)
    setIsLoading(false)
  }, [])

  return {
    messages,
    isLoading,
    agentState,
    resultData,
    error,
    sendMessage,
    resetSession,
    sessionId: sessionIdRef.current,
  }
}
