'use client'

import { createContext, useContext, useState, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { apiService } from '@/lib/apiService'

const AISheetChatContext = createContext({})

export const useAISheetChat = () => {
  const context = useContext(AISheetChatContext)
  if (!context) {
    throw new Error('useAISheetChat must be used within AISheetChatProvider')
  }
  return context
}

export const AISheetChatProvider = ({ children }) => {
  const [currentChatId, setCurrentChatId] = useState(null)
  const [datasetLoaded, setDatasetLoaded] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [messages, setMessages] = useState([])
  const [chats, setChats] = useState([])
  const [datasetInfo, setDatasetInfo] = useState(null)
  const [insights, setInsights] = useState([])
  const [progress, setProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const statusIntervalRef = useRef(null)

  useEffect(() => {
    initializeApp()
    
    // Cleanup on unmount
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current)
      }
    }
  }, [])

  const initializeApp = async () => {
    await loadChatHistory()
    
    // Check if there's an active dataset on app load
    try {
      const status = await apiService.getStatus()
      if (status.stage === 'complete') {
        setDatasetLoaded(true)
        setDatasetInfo(status.dataset_info)
        setInsights(status.insights || [])
      }
    } catch (error) {
      console.log('No active dataset on startup')
    }
  }

  const createNewChat = async () => {
    try {
      // Clear current state for new chat
      setMessages([])
      // Don't clear dataset state - keep it persistent
      setUploadError(null)
      setProgress(0)
      
      const result = await apiService.createNewChat()
      if (result.success) {
        setCurrentChatId(result.chat_id)
        
        // If we have a dataset loaded, add welcome message to new chat
        if (datasetLoaded && datasetInfo) {
          const welcomeMessage = {
            id: `welcome-${Date.now()}`,
            type: 'assistant',
            content: `🧠 **AI Analysis Ready!**\n\n` +
              `I've analyzed your dataset with **${datasetInfo.shape.rows.toLocaleString()} rows** and **${datasetInfo.shape.columns} columns**.\n\n` +
              `📊 **File size:** ${datasetInfo.file_size_formatted || `${datasetInfo.file_size_mb} MB`}\n\n` +
              `✨ **I can help you with:**\n` +
              `• Natural language questions about your data\n` +
              `• Pattern recognition and insights\n` +
              `• Statistical analysis and summaries\n` +
              `• Data filtering and visualization\n` +
              `• Anomaly detection\n` +
              `• Recommendations based on data\n\n` +
              `Ask me anything! I understand natural language and provide complete results.`,
            analysisType: 'ai_analysis',
            timestamp: new Date().toISOString()
          }
          setMessages([welcomeMessage])
        }
        
        // Reload chat history to get the updated list
        setTimeout(async () => {
          await loadChatHistory()
        }, 100)
        
        toast.success('New analysis session created')
      }
    } catch (error) {
      console.error('Failed to create chat:', error)
      toast.error('Failed to create new chat')
    }
  }

  const loadChatHistory = async () => {
    try {
      const result = await apiService.getChatHistory()
      setChats(result.chats || [])
      
      // If no current chat and we have chats, select the first one
      if (!currentChatId && result.chats && result.chats.length > 0) {
        setCurrentChatId(result.chats[0].id)
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }

  const switchToChat = async (chatId) => {
    if (chatId === currentChatId) return // Don't switch if already current
    
    try {
      const result = await apiService.getChat(chatId)
      if (result.chat) {
        setCurrentChatId(chatId)
        
        const formattedMessages = result.chat.messages.map((message, index) => ({
          id: `${chatId}-${index}-${Date.now()}`,
          type: message.type,
          content: message.content,
          query: message.query,
          result: message.result,
          analysisType: message.type === 'assistant' ? 'ai_analysis' : null,
          timestamp: message.timestamp
        }))
        
        setMessages(formattedMessages)
        
        // Check if this chat has dataset loaded by checking welcome message
        const hasWelcomeMessage = formattedMessages.some(msg => 
          msg.type === 'assistant' && msg.content.includes('AI Analysis Ready')
        )
        
        if (hasWelcomeMessage) {
          // Try to restore dataset state
          try {
            const status = await apiService.getStatus()
            if (status.stage === 'complete') {
              setDatasetLoaded(true)
              setDatasetInfo(status.dataset_info)
              setInsights(status.insights || [])
            } else {
              // Check if there's a dataset available for this chat
              const datasetStatus = await apiService.getChatDatasetStatus(chatId)
              if (datasetStatus.has_dataset) {
                setDatasetLoaded(true)
                setDatasetInfo(datasetStatus.dataset_info)
                setInsights([])
              }
            }
          } catch (error) {
            console.log('Could not restore dataset state for this chat')
          }
        } else {
          // No welcome message - check if we should keep current dataset state
          // Only clear if no dataset is available globally
          try {
            const status = await apiService.getStatus()
            if (status.stage !== 'complete') {
              setDatasetLoaded(false)
              setDatasetInfo(null)
              setInsights([])
            }
          } catch (error) {
            // Keep current state if we can't check
          }
        }
        
        await loadChatHistory()
      }
    } catch (error) {
      console.error('Failed to switch chat:', error)
      toast.error('Failed to load chat')
    }
  }

  const deleteChat = async (chatId) => {
    try {
      const result = await apiService.deleteChat(chatId)
      if (result.success) {
        toast.success('Chat deleted successfully')
        
        // If we deleted the current chat
        if (chatId === currentChatId) {
          // Find another chat to switch to
          const remainingChats = chats.filter(chat => chat.id !== chatId)
          
          if (remainingChats.length > 0) {
            // Switch to the first remaining chat
            await switchToChat(remainingChats[0].id)
          } else {
            // No chats left - clear messages but keep dataset if available
            setCurrentChatId(null)
            setMessages([])
            // Don't clear dataset state - it persists across sessions
          }
        }
        
        // Always reload chat history
        await loadChatHistory()
      } else {
        toast.error('Failed to delete chat')
      }
    } catch (error) {
      console.error('Failed to delete chat:', error)
      toast.error('Failed to delete chat')
    }
  }

  const renameChat = async (chatId, newTitle) => {
    try {
      const result = await apiService.renameChat(chatId, newTitle)
      if (result.success) {
        toast.success('Chat renamed successfully')
        await loadChatHistory()
      } else {
        toast.error(result.error || 'Failed to rename chat')
      }
    } catch (error) {
      console.error('Failed to rename chat:', error)
      toast.error('Failed to rename chat')
    }
  }

  const uploadFile = async (file) => {
    if (!file) return

    const allowedTypes = ['.csv', '.xlsx', '.xls']
    const fileExt = '.' + file.name.split('.').pop().toLowerCase()
    
    if (!allowedTypes.includes(fileExt)) {
      toast.error('Please upload a CSV or Excel file')
      return
    }

    if (file.size > 10 * 1024 * 1024 * 1024) {
      toast.error('File too large. Maximum size is 10GB')
      return
    }

    setIsUploading(true)
    setProgress(0)
    setUploadError(null)

    try {
      toast.loading(`Uploading ${file.name}...`, { id: 'upload-progress' })
      
      const result = await apiService.uploadFile(file)
      
      if (result.success) {
        toast.success(`Processing ${file.name}...`, { id: 'upload-progress' })
        startStatusMonitoring()
      } else {
        toast.error(result.error || 'Upload failed', { id: 'upload-progress' })
        setIsUploading(false)
        setUploadError(result.error)
      }
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Upload failed: ' + error.message, { id: 'upload-progress' })
      setIsUploading(false)
      setUploadError(error.message)
    }
  }

  const startStatusMonitoring = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current)
    }

    statusIntervalRef.current = setInterval(async () => {
      try {
        const status = await apiService.getStatus()
        setProgress(status.progress || 0)
        
        if (status.stage === 'complete') {
          clearInterval(statusIntervalRef.current)
          statusIntervalRef.current = null
          onDatasetLoaded(status)
        } else if (status.stage === 'error') {
          clearInterval(statusIntervalRef.current)
          statusIntervalRef.current = null
          toast.error(status.error || 'Processing failed', { id: 'upload-progress' })
          setIsUploading(false)
          setUploadError(status.error)
        }
      } catch (error) {
        console.error('Status check failed:', error)
        clearInterval(statusIntervalRef.current)
        statusIntervalRef.current = null
        setIsUploading(false)
        toast.error('Status check failed', { id: 'upload-progress' })
      }
    }, 1000)
  }

  const onDatasetLoaded = (status) => {
    setDatasetLoaded(true)
    setIsUploading(false)
    setProgress(100)
    setDatasetInfo(status.dataset_info)
    setInsights(status.insights || [])
    setUploadError(null)
    
    const welcomeMessage = {
      id: `welcome-${Date.now()}`,
      type: 'assistant',
      content: `🧠 **AI Analysis Ready!**\n\n` +
        `I've analyzed your dataset with **${status.dataset_info.shape.rows.toLocaleString()} rows** and **${status.dataset_info.shape.columns} columns**.\n\n` +
        `📊 **File size:** ${status.dataset_info.file_size_formatted || `${status.dataset_info.file_size_mb} MB`}\n\n` +
        `✨ **I can help you with:**\n` +
        `• Natural language questions about your data\n` +
        `• Pattern recognition and insights\n` +
        `• Statistical analysis and summaries\n` +
        `• Data filtering and visualization\n` +
        `• Anomaly detection\n` +
        `• Recommendations based on data\n\n` +
        `Ask me anything! I understand natural language and provide complete results.`,
      analysisType: 'ai_analysis',
      timestamp: new Date().toISOString()
    }
    
    setMessages([welcomeMessage])
    toast.success('🎉 Dataset ready! Start asking questions about your data.', { id: 'upload-progress' })
  }

  const sendMessage = async (message) => {
    if (!datasetLoaded || isProcessing || !message.trim()) return

    setIsProcessing(true)
    
    const userMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: message.trim(),
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMessage])

    try {
      const result = await apiService.sendQuery(message.trim(), currentChatId)
      
      if (result.success) {
        // Check if this is a visualization request
        const isVisualizationRequest = message.toLowerCase().includes('visualiz') || 
                                     message.toLowerCase().includes('chart') || 
                                     message.toLowerCase().includes('graph') || 
                                     message.toLowerCase().includes('plot')
        
        const assistantMessage = {
          id: `assistant-${Date.now()}`,
          type: 'assistant',
          content: result.answer,
          query: result.sql_query,
          result: result.results,
          analysisType: result.analysis_type || 'ai_analysis',
          chartData: isVisualizationRequest ? generateChartData(result.results) : null,
          timestamp: new Date().toISOString()
        }
        
        setMessages(prev => [...prev, assistantMessage])
        await loadChatHistory()
      } else {
        const errorMessage = {
          id: `error-${Date.now()}`,
          type: 'assistant',
          content: result.error || 'I encountered an issue analyzing your data. Please try rephrasing your question.',
          timestamp: new Date().toISOString()
        }
        
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Query failed:', error)
      const errorMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: 'I encountered a technical issue. Please try again or rephrase your question.',
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsProcessing(false)
    }
  }

  const generateChartData = (data) => {
    if (!data || !Array.isArray(data) || data.length === 0) return null
    
    try {
      const headers = Object.keys(data[0])
      const numericColumns = headers.filter(header => 
        data.some(row => typeof row[header] === 'number' && !isNaN(row[header]))
      )
      
      if (numericColumns.length === 0) return null
      
      // Create a simple bar chart data
      const chartData = {
        type: 'bar',
        data: {
          labels: data.slice(0, 10).map((row, index) => 
            row.name || row.label || row.category || `Item ${index + 1}`
          ),
          datasets: [{
            label: numericColumns[0],
            data: data.slice(0, 10).map(row => row[numericColumns[0]] || 0),
            backgroundColor: 'rgba(133, 0, 223, 0.8)',
            borderColor: 'rgba(133, 0, 223, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Data Visualization'
            }
          }
        }
      }
      
      return chartData
    } catch (error) {
      console.error('Error generating chart data:', error)
      return null
    }
  }

  const clearChat = async () => {
    setMessages([])
    // Don't clear dataset state - it persists
    // If we have a dataset loaded, add welcome message back
    if (datasetLoaded && datasetInfo) {
      const welcomeMessage = {
        id: `welcome-${Date.now()}`,
        type: 'assistant',
        content: `🧠 **AI Analysis Ready!**\n\n` +
          `I've analyzed your dataset with **${datasetInfo.shape.rows.toLocaleString()} rows** and **${datasetInfo.shape.columns} columns**.\n\n` +
          `📊 **File size:** ${datasetInfo.file_size_formatted || `${datasetInfo.file_size_mb} MB`}\n\n` +
          `✨ **I can help you with:**\n` +
          `• Natural language questions about your data\n` +
          `• Pattern recognition and insights\n` +
          `• Statistical analysis and summaries\n` +
          `• Data filtering and visualization\n` +
          `• Anomaly detection\n` +
          `• Recommendations based on data\n\n` +
          `Ask me anything! I understand natural language and provide complete results.`,
        analysisType: 'ai_analysis',
        timestamp: new Date().toISOString()
      }
      setMessages([welcomeMessage])
    }
  }

  const value = {
    currentChatId,
    datasetLoaded,
    isProcessing,
    messages,
    chats,
    datasetInfo,
    insights,
    progress,
    isUploading,
    uploadError,
    createNewChat,
    loadChatHistory,
    switchToChat,
    deleteChat,
    renameChat,
    uploadFile,
    sendMessage,
    clearChat,
    setMessages
  }

  return (
    <AISheetChatContext.Provider value={value}>
      {children}
    </AISheetChatContext.Provider>
  )
}