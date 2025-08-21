'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Progress } from '@/components/ui/progress'
import { 
  Brain, 
  Plus, 
  Upload, 
  Database, 
  BarChart3,
  Trash2,
  FileText,
  AlertCircle,
  CheckCircle,
  MessageSquare,
  Clock,
  PieChart,
  Activity,
  Edit2,
  MoreVertical
} from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { useAISheetChat } from '@/contexts/AISheetChatContext'
import { formatTime } from '@/lib/utils'
import { cn } from '@/lib/utils'
import DeleteChatDialog from '@/components/DeleteChatDialog'
import RenameChatDialog from '@/components/RenameChatDialog'

export default function Sidebar() {
  const {
    currentChatId,
    datasetLoaded,
    chats,
    datasetInfo,
    progress,
    isUploading,
    uploadError,
    createNewChat,
    switchToChat,
    deleteChat,
    renameChat,
    uploadFile
  } = useAISheetChat()

  // Dialog states
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [chatToDelete, setChatToDelete] = useState(null)
  const [renameDialogOpen, setRenameDialogOpen] = useState(false)
  const [chatToRename, setChatToRename] = useState(null)
  const [openDropdownId, setOpenDropdownId] = useState(null)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      return
    }
    if (acceptedFiles.length > 0) {
      uploadFile(acceptedFiles[0])
    }
  }, [uploadFile])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false,
    disabled: isUploading || !currentChatId,
    maxSize: 10 * 1024 * 1024 * 1024,
  })

  // Dropdown handlers
  const toggleDropdown = (chatId, event) => {
    event.preventDefault()
    event.stopPropagation()
    setOpenDropdownId(openDropdownId === chatId ? null : chatId)
  }

  const closeDropdown = () => {
    setOpenDropdownId(null)
  }

  // Delete handlers
  const handleDeleteClick = (chat, event) => {
    event.preventDefault()
    event.stopPropagation()
    setChatToDelete(chat)
    setDeleteDialogOpen(true)
    closeDropdown()
  }

  const handleDeleteConfirm = async () => {
    if (chatToDelete) {
      try {
        await deleteChat(chatToDelete.id)
      } catch (error) {
        console.error('Error deleting chat:', error)
      } finally {
        setChatToDelete(null)
      }
    }
  }

  const handleDeleteCancel = () => {
    setChatToDelete(null)
    setDeleteDialogOpen(false)
  }

  // Rename handlers
  const handleRenameClick = (chat, event) => {
    event.preventDefault()
    event.stopPropagation()
    setChatToRename(chat)
    setRenameDialogOpen(true)
    closeDropdown()
  }

  const handleRenameConfirm = async (newTitle) => {
    if (chatToRename && newTitle.trim()) {
      try {
        await renameChat(chatToRename.id, newTitle.trim())
      } catch (error) {
        console.error('Error renaming chat:', error)
        throw error
      }
    }
  }

  const handleRenameCancel = () => {
    setChatToRename(null)
    setRenameDialogOpen(false)
  }

  // File upload handler
  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadFile(file)
    }
  }

  const getDatasetStats = () => {
    if (!datasetInfo) return []
    
    return [
      { 
        icon: Database, 
        label: 'Rows', 
        value: datasetInfo.shape.rows.toLocaleString()
      },
      { 
        icon: BarChart3, 
        label: 'Columns', 
        value: datasetInfo.shape.columns.toString()
      },
      { 
        icon: FileText, 
        label: 'Size', 
        value: datasetInfo.file_size_formatted || `${(datasetInfo.file_size_mb || 0).toFixed(1)} MB`
      },
    ]
  }

  // Utility function for truncating text
  const truncateText = (text, maxLength) => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength) + '...'
  }

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      closeDropdown()
    }

    if (openDropdownId) {
      setTimeout(() => {
        document.addEventListener('click', handleClickOutside)
      }, 100)
      
      return () => {
        document.removeEventListener('click', handleClickOutside)
      }
    }
  }, [openDropdownId])

  return (
    <>
      <div className="w-80 bg-card border-r border-border flex flex-col h-full shadow-sm relative">
        {/* Header - Fixed Height to Match Navbar */}
        <div className="h-[120px] flex flex-col justify-center p-6" style={{ backgroundColor: '#8500df', color: 'white' }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
              <Brain className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">AI Sheet Chat</h1>
            </div>
          </div>
          
          <Button 
            onClick={createNewChat}
            className="w-full bg-white/20 hover:bg-white/30 text-white border-white/20 backdrop-blur-sm font-medium"
            variant="outline"
            disabled={isUploading}
          >
            <Plus className="h-4 w-4 mr-2" />
            New Analysis Session
          </Button>
        </div>

        {/* Upload Section */}
        <div className="p-6 border-b border-border">
          <div className="flex items-center gap-2 mb-4">
            <Upload className="h-4 w-4 text-gray-600" />
            <h3 className="text-sm font-semibold text-gray-800">Data Upload</h3>
            {!currentChatId && (
              <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-700">
                Select chat first
              </Badge>
            )}
          </div>
          
          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all",
              "hover:border-primary hover:bg-primary/5",
              isDragActive && !isDragReject && "border-primary bg-primary/5",
              isDragReject && "border-red-300 bg-red-50",
              (isUploading || !currentChatId) && "opacity-60 cursor-not-allowed",
              uploadError && "border-red-300 bg-red-50"
            )}
          >
            <input 
              {...getInputProps()} 
              onChange={handleFileInputChange}
              disabled={isUploading || !currentChatId}
            />
            
            <div className="relative z-10">
              {!currentChatId ? (
                <div className="space-y-2">
                  <MessageSquare className="h-8 w-8 mx-auto text-gray-400" />
                  <div className="text-sm font-medium text-gray-700">Create or select a chat first</div>
                  <div className="text-xs text-gray-500">Then upload your data file</div>
                </div>
              ) : isUploading ? (
                <div className="space-y-3">
                  <Database className="h-8 w-8 mx-auto text-purple-600 animate-pulse" />
                  <div className="text-sm font-medium text-gray-700">Processing your file...</div>
                  <Progress value={progress} className="h-2" />
                  <div className="text-xs text-gray-500">{progress}% complete</div>
                </div>
              ) : uploadError ? (
                <div className="space-y-2">
                  <AlertCircle className="h-8 w-8 mx-auto text-red-500" />
                  <div className="text-sm font-medium text-red-700">Upload failed</div>
                  <div className="text-xs text-red-500">{uploadError}</div>
                </div>
              ) : datasetLoaded ? (
                <div className="space-y-2">
                  <CheckCircle className="h-8 w-8 mx-auto text-green-500" />
                  <div className="text-sm font-medium text-green-700">Dataset ready for this chat</div>
                  <div className="text-xs text-green-600">Click to upload new data</div>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="h-8 w-8 mx-auto text-gray-400" />
                  <div className="text-sm font-medium text-gray-700">
                    {isDragActive ? 'Drop your file here' : 'Drop files or click to browse'}
                  </div>
                  <div className="text-xs text-gray-500">CSV, Excel files up to 10GB</div>
                </div>
              )}
            </div>
          </div>

          {/* Dataset Stats */}
          {datasetLoaded && datasetInfo && (
            <Card className="mt-4 border-0 shadow-sm" style={{ background: 'linear-gradient(135deg, #f5e7ff 0%, #e8d5ff 100%)' }}>
              <CardContent className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <PieChart className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-semibold text-gray-800">Dataset Overview</span>
                  <Badge variant="secondary" className="text-xs bg-green-100 text-green-800 ml-auto">
                    Active
                  </Badge>
                </div>
                
                <div className="grid grid-cols-1 gap-3">
                  {getDatasetStats().map((stat, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white/70 rounded-lg backdrop-blur-sm">
                      <div className="flex items-center gap-2">
                        <stat.icon className="h-3 w-3 text-purple-600" />
                        <span className="text-xs text-gray-600">{stat.label}</span>
                      </div>
                      <Badge variant="secondary" className="text-xs font-medium bg-white/80">
                        {stat.value}
                      </Badge>
                    </div>
                  ))}
                </div>
                
                <div className="mt-3 p-2 bg-green-100 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Activity className="h-3 w-3 text-green-600" />
                    <span className="text-xs font-medium text-green-800">Ready for Analysis</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Chat History - WORKING TRUNCATION + VISIBLE 3-DOTS */}
        <div className="flex-1 flex flex-col min-h-0 relative">
          <div className="p-6 pb-4 flex-shrink-0">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-gray-600" />
              <h3 className="text-sm font-semibold text-gray-800">Recent Sessions</h3>
              <Badge variant="secondary" className="text-xs">{chats.length}</Badge>
            </div>
          </div>
          
          {/* Scrollable Chat List */}
          <div className="flex-1 min-h-0 px-6 relative">
            <ScrollArea className="h-full">
              <div className="space-y-2 pb-6">
                {chats.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageSquare className="h-8 w-8 mx-auto text-gray-300 mb-2" />
                    <div className="text-sm text-gray-500">No sessions yet</div>
                    <div className="text-xs text-gray-400">Create a new session to start</div>
                  </div>
                ) : (
                  chats.map((chat) => {
                    const lastMessage = chat.messages && chat.messages.length > 0 
                      ? chat.messages[chat.messages.length - 1] 
                      : null
                    const preview = lastMessage 
                      ? (lastMessage.type === 'user' ? lastMessage.content : 'AI Analysis')
                      : 'New Session'
                    
                    // MANUAL TRUNCATION - GUARANTEED TO WORK
                    const truncatedTitle = truncateText(chat.title, 25)
                    const truncatedPreview = truncateText(preview, 35)
                    
                    return (
                      <div
                        key={chat.id}
                        className={cn(
                          "relative rounded-lg border transition-all",
                          "hover:shadow-md hover:bg-gray-50",
                          chat.id === currentChatId 
                            ? "bg-purple-50 border-purple-200 shadow-sm" 
                            : "bg-white border-gray-100 hover:border-gray-200"
                        )}
                      >
                        {/* GRID LAYOUT - FIXED PROPORTIONS */}
                        <div className="p-3">
                          <div 
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1fr 32px', // Content takes remaining space, 32px for button
                              gap: '8px',
                              alignItems: 'start'
                            }}
                          >
                            {/* LEFT COLUMN: Chat Content */}
                            <div 
                              className="cursor-pointer overflow-hidden"
                              onClick={() => switchToChat(chat.id)}
                              style={{ minWidth: 0 }} // Critical for truncation
                            >
                              {/* Title with indicator */}
                              <div className="flex items-center gap-2 mb-2">
                                <div className={cn(
                                  "w-2 h-2 rounded-full flex-shrink-0",
                                  chat.id === currentChatId ? "bg-purple-500" : "bg-gray-300"
                                )} />
                                <div 
                                  className="font-medium text-sm text-gray-900"
                                  style={{
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    whiteSpace: 'nowrap'
                                  }}
                                  title={chat.title} // Show full title on hover
                                >
                                  {truncatedTitle}
                                </div>
                              </div>
                              
                              {/* Preview text */}
                              <div 
                                className="text-xs text-gray-500 mb-2"
                                style={{
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap'
                                }}
                                title={preview} // Show full preview on hover
                              >
                                {truncatedPreview}
                              </div>
                              
                              {/* Bottom metadata */}
                              <div className="flex items-center justify-between text-xs text-gray-400">
                                <div 
                                  className="flex items-center gap-1"
                                  style={{ minWidth: 0 }}
                                >
                                  <Clock className="h-3 w-3 flex-shrink-0" />
                                  <span 
                                    style={{
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap'
                                    }}
                                  >
                                    {formatTime(chat.updated_at)}
                                  </span>
                                </div>
                                {chat.messages && chat.messages.length > 0 && (
                                  <div className="flex items-center gap-1 flex-shrink-0">
                                    <MessageSquare className="h-3 w-3" />
                                    <span>{chat.messages.length}</span>
                                  </div>
                                )}
                              </div>
                            </div>
                            
                            {/* RIGHT COLUMN: 3-Dots Button */}
                            <div className="relative flex justify-center">
                              <button
                                onClick={(e) => toggleDropdown(chat.id, e)}
                                className={cn(
                                  "w-8 h-8 rounded-md flex items-center justify-center",
                                  "text-gray-500 hover:text-gray-700 hover:bg-gray-200",
                                  "transition-colors duration-200",
                                  openDropdownId === chat.id && "bg-gray-200 text-gray-700"
                                )}
                                title="More options"
                              >
                                <MoreVertical className="h-4 w-4" />
                              </button>
                              
                              {/* DROPDOWN MENU */}
                              {openDropdownId === chat.id && (
                                <div 
                                  className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-[140px]"
                                  onClick={(e) => e.stopPropagation()}
                                  style={{
                                    zIndex: 9999,
                                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
                                  }}
                                >
                                  {/* Rename Option */}
                                  <button
                                    onClick={(e) => handleRenameClick(chat, e)}
                                    className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3 transition-colors"
                                  >
                                    <Edit2 className="h-4 w-4 text-blue-500" />
                                    <span>Rename</span>
                                  </button>
                                  
                                  {/* Delete Option */}
                                  <button
                                    onClick={(e) => handleDeleteClick(chat, e)}
                                    className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-3 transition-colors"
                                  >
                                    <Trash2 className="h-4 w-4 text-red-500" />
                                    <span>Delete</span>
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>

      {/* Delete Dialog */}
      <DeleteChatDialog
        open={deleteDialogOpen}
        onOpenChange={(open) => {
          setDeleteDialogOpen(open)
          if (!open) {
            setChatToDelete(null)
          }
        }}
        onConfirm={handleDeleteConfirm}
        chatTitle={chatToDelete?.title || ''}
      />

      {/* Rename Dialog */}
      <RenameChatDialog
        open={renameDialogOpen}
        onOpenChange={(open) => {
          setRenameDialogOpen(open)
          if (!open) {
            setChatToRename(null)
          }
        }}
        onConfirm={handleRenameConfirm}
        currentTitle={chatToRename?.title || ''}
      />
    </>
  )
}