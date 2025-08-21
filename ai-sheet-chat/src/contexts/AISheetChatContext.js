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
    
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current)
      }
    }
  }, [])

  const initializeApp = async () => {
    await loadChatHistory()
  }

  const createNewChat = async () => {
    try {
      setMessages([])
      setDatasetLoaded(false)
      setDatasetInfo(null)
      setInsights([])
      setUploadError(null)
      setProgress(0)
      
      const result = await apiService.createNewChat()
      if (result.success) {
        setCurrentChatId(result.chat_id)
        
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
      
      if (!currentChatId && result.chats && result.chats.length > 0) {
        setCurrentChatId(result.chats[0].id)
        await loadChatDataset(result.chats[0].id)
      }
    } catch (error) {
      console.error('Failed to load chat history:', error)
    }
  }

  const loadChatDataset = async (chatId) => {
    try {
      const datasetStatus = await apiService.getChatDatasetStatus(chatId)
      if (datasetStatus.has_dataset) {
        setDatasetLoaded(true)
        setDatasetInfo(datasetStatus.dataset_info)
        
        // Get insights from chat status
        try {
          const status = await apiService.getChatStatus(chatId)
          if (status.stage === 'complete') {
            setInsights(status.insights || [])
          }
        } catch (error) {
          console.log('Could not load insights for this chat')
        }
      } else {
        setDatasetLoaded(false)
        setDatasetInfo(null)
        setInsights([])
      }
    } catch (error) {
      console.log('No dataset for this chat')
      setDatasetLoaded(false)
      setDatasetInfo(null)
      setInsights([])
    }
  }

  const switchToChat = async (chatId) => {
    if (chatId === currentChatId) return
    
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
        
        // Load dataset for this specific chat
        await loadChatDataset(chatId)
        
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
        
        if (chatId === currentChatId) {
          const remainingChats = chats.filter(chat => chat.id !== chatId)
          
          if (remainingChats.length > 0) {
            await switchToChat(remainingChats[0].id)
          } else {
            setCurrentChatId(null)
            setMessages([])
            setDatasetLoaded(false)
            setDatasetInfo(null)
            setInsights([])
          }
        }
        
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
    if (!file || !currentChatId) {
      toast.error('Please create or select a chat first')
      return
    }

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
      
      const result = await apiService.uploadFileForChat(file, currentChatId)
      
      if (result.success) {
        toast.success(`Dataset loaded successfully for this chat!`, { id: 'upload-progress' })
        setIsUploading(false)
        setProgress(100)
        
        // Reload dataset status for current chat
        await loadChatDataset(currentChatId)
        
        // Add welcome message
        const welcomeMessage = {
          id: `welcome-${Date.now()}`,
          type: 'assistant',
          content: `🧠 **AI Analysis Ready!**\n\nI've analyzed your dataset and it's ready for questions!\n\nAsk me anything about your data - I understand natural language and provide complete results.`,
          analysisType: 'ai_analysis',
          timestamp: new Date().toISOString()
        }
        
        setMessages([welcomeMessage])
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

const sendMessage = async (message) => {
  if (!datasetLoaded || isProcessing || !message.trim() || !currentChatId) return

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
      // Handle specific error types
      let errorMessage = result.error || 'I encountered an issue analyzing your data.'
      let showSpecialError = false
      
      if (result.error_type === 'RATE_LIMIT_EXCEEDED') {
        errorMessage = `🚫 **API Rate Limit Exceeded**

${result.user_message}

**What happened?**
The Gemini AI service has reached its rate limit for your current API key.

**Solutions:**
1. **Wait and retry** - Rate limits reset after a few minutes
2. **Upgrade API quota** - Contact your administrator to upgrade the Gemini API plan
3. **Use a different API key** - If you have access to another API key with higher limits

**Technical details:** ${result.error}`
        showSpecialError = true
        toast.error('API Rate Limit Exceeded - Please wait or upgrade quota')
      } else if (result.error_type === 'GEMINI_API_ERROR') {
        errorMessage = `⚠️ **AI Service Error**

${result.user_message}

**Technical details:** ${result.error}

Please try again in a moment. If the issue persists, contact support.`
        toast.error('AI Service Error - Please try again')
      } else if (result.error_type === 'GEMINI_API_TIMEOUT') {
        errorMessage = `⏱️ **Request Timeout**

${result.user_message}

The AI service is experiencing high load. Please try again.`
        toast.error('Request timed out - Please try again')
      } else if (result.error_type === 'GEMINI_API_CONNECTION') {
        errorMessage = `🌐 **Connection Error**

${result.user_message}

Please check your internet connection and try again.`
        toast.error('Connection error - Check internet connection')
      } else {
        toast.error('Analysis failed - Please try again')
      }
      
      const errorMessageObj = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: errorMessage,
        error_type: result.error_type,
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, errorMessageObj])
    }
  } catch (error) {
    console.error('Query failed:', error)
    const errorMessage = {
      id: `error-${Date.now()}`,
      type: 'assistant',
      content: `🔧 **Technical Issue**

I encountered a technical issue while processing your request.

**Error:** ${error.message}

Please try again or contact support if the issue persists.`,
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, errorMessage])
    toast.error('Technical error occurred')
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
    
    // If we have a dataset loaded for this chat, add welcome message back
    if (datasetLoaded && datasetInfo) {
      const welcomeMessage = {
        id: `welcome-${Date.now()}`,
        type: 'assistant',
        content: `🧠 **AI Analysis Ready!**\n\nI've analyzed your dataset and it's ready for questions!\n\nAsk me anything about your data - I understand natural language and provide complete results.`,
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