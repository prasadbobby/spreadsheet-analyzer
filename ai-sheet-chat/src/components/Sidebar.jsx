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
  MoreVertical,
  History,
  Menu,
  X,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { useAISheetChat } from '@/contexts/AISheetChatContext'
import { useSidebarContext } from '@/contexts/SidebarContext'
import { formatTime } from '@/lib/utils'
import { cn } from '@/lib/utils'
import DeleteChatDialog from '@/components/DeleteChatDialog'
import RenameChatDialog from '@/components/RenameChatDialog'
import RecentChatsModal from '@/components/RecentChatsModal'

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

  const { isCollapsed, isMobile, toggleSidebar, closeSidebar } = useSidebarContext()

  // Dialog states
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [chatToDelete, setChatToDelete] = useState(null)
  const [renameDialogOpen, setRenameDialogOpen] = useState(false)
  const [chatToRename, setChatToRename] = useState(null)
  const [openDropdownId, setOpenDropdownId] = useState(null)
  const [recentChatsModalOpen, setRecentChatsModalOpen] = useState(false)

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

  // Get current chat
  const currentChat = chats.find(chat => chat.id === currentChatId)

  // Dropdown handlers for current chat
  const toggleDropdown = (event) => {
    event.preventDefault()
    event.stopPropagation()
    setOpenDropdownId(openDropdownId === currentChatId ? null : currentChatId)
  }

  const closeDropdown = () => {
    setOpenDropdownId(null)
  }

  // Delete handlers
  const handleDeleteClick = (event) => {
    event.preventDefault()
    event.stopPropagation()
    if (currentChat) {
      setChatToDelete(currentChat)
      setDeleteDialogOpen(true)
      closeDropdown()
    }
  }

  const handleDeleteFromModal = (chatId, chatTitle) => {
    const chat = chats.find(c => c.id === chatId)
    if (chat) {
      setChatToDelete({ id: chatId, title: chatTitle })
      setDeleteDialogOpen(true)
    }
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
  const handleRenameClick = (event) => {
    event.preventDefault()
    event.stopPropagation()
    if (currentChat) {
      setChatToRename(currentChat)
      setRenameDialogOpen(true)
      closeDropdown()
    }
  }

  const handleRenameFromModal = (chatId, currentTitle) => {
    setChatToRename({ id: chatId, title: currentTitle })
    setRenameDialogOpen(true)
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

  // Sidebar width calculation
  const sidebarWidth = isCollapsed ? 'w-16' : 'w-80'

  return (
    <>
      {/* Mobile Overlay */}
      {isMobile && !isCollapsed && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={closeSidebar}
        />
      )}

      {/* Sidebar */}
      <div className={cn(
        "bg-card border-r border-border flex flex-col h-full shadow-sm relative transition-all duration-300 ease-in-out z-50",
        sidebarWidth
      )}>
        
        {/* Collapse Toggle Button - KEEP THIS RELATIVE TO SIDEBAR */}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className={cn(
            "absolute -right-3 top-4 z-10 h-6 w-6 rounded-full border bg-white shadow-md hover:shadow-lg transition-all",
            "flex items-center justify-center p-0"
          )}
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>

        {/* Header */}
        <div 
          className={cn(
            "flex flex-col justify-center transition-all duration-300",
            isCollapsed ? "h-16 p-3" : "h-[120px] p-6"
          )} 
          style={{ backgroundColor: '#8500df', color: 'white' }}
        >
          {isCollapsed ? (
            // Collapsed Header
            <div className="flex items-center justify-center">
              <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
                <Brain className="h-6 w-6" />
              </div>
            </div>
          ) : (
            // Full Header
            <>
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
            </>
          )}
        </div>

        {/* Collapsed Navigation */}
        {isCollapsed ? (
          <div className="flex-1 flex flex-col items-center p-3 space-y-4">
            {/* Upload Icon */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => !isCollapsed && null}
              className="w-10 h-10 p-0 rounded-lg hover:bg-gray-100"
              title="Upload Data"
            >
              <Upload className="h-5 w-5 text-gray-600" />
            </Button>

            {/* Dataset Status */}
            {datasetLoaded ? (
              <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
                <Database className="h-5 w-5 text-gray-400" />
              </div>
            )}

            {/* Current Chat Icon */}
            {currentChat && (
              <div className="w-10 h-10 rounded-lg bg-purple-100 flex items-center justify-center">
                <MessageSquare className="h-5 w-5 text-purple-600" />
              </div>
            )}

            {/* Recent Sessions */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setRecentChatsModalOpen(true)}
              className="w-10 h-10 p-0 rounded-lg hover:bg-gray-100 relative"
              title="Recent Sessions"
            >
              <History className="h-5 w-5 text-gray-600" />
              {chats.length > 0 && (
                <Badge 
                  variant="secondary" 
                  className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs bg-purple-100 text-purple-800 flex items-center justify-center"
                >
                  {chats.length > 9 ? '9+' : chats.length}
                </Badge>
              )}
            </Button>
          </div>
        ) : (
          // Full Content
          <>
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

            {/* Current Chat Section */}
            <div className="flex-1 flex flex-col min-h-0 relative">
              {/* Header with Recent Sessions Button */}
              <div className="p-6 pb-4 flex-shrink-0">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-gray-600" />
                    <h3 className="text-sm font-semibold text-gray-800">Current Session</h3>
                  </div>
                  
                  {/* Recent Sessions Button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setRecentChatsModalOpen(true)}
                    className="flex items-center gap-2 text-xs hover:bg-gray-100 px-3 py-2 h-auto"
                    title="View all recent sessions"
                  >
                    <History className="h-4 w-4" />
                    <span className="font-medium">Recent Sessions</span>
                    <Badge variant="secondary" className="text-xs ml-1">
                      {chats.length}
                    </Badge>
                  </Button>
                </div>
              </div>
              
              {/* Current Chat Display */}
              <div className="flex-1 min-h-0 px-6 relative">
                <div className="pb-6">
                  {!currentChat ? (
                    <div className="text-center py-8">
                      <MessageSquare className="h-8 w-8 mx-auto text-gray-300 mb-2" />
                      <div className="text-sm text-gray-500">No active session</div>
                      <div className="text-xs text-gray-400 mb-4">Create a new session to start</div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setRecentChatsModalOpen(true)}
                        className="text-sm"
                      >
                        <History className="h-4 w-4 mr-2" />
                        Browse Sessions
                      </Button>
                    </div>
                  ) : (
                    <div
                      className={cn(
                        "relative rounded-lg border transition-all",
                        "bg-purple-50 border-purple-200 shadow-sm"
                      )}
                    >
                      <div className="p-4">
                        <div 
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 32px',
                            gap: '8px',
                            alignItems: 'start'
                          }}
                        >
                          {/* Chat Content */}
                          <div 
                            className="overflow-hidden"
                            style={{ minWidth: 0 }}
                          >
                            {/* Title with indicator */}
                            <div className="flex items-center gap-2 mb-3">
                              <div className="w-3 h-3 bg-purple-500 rounded-full flex-shrink-0 animate-pulse" />
                              <div className="font-medium text-sm text-gray-900">
                                Active Session
                              </div>
                            </div>

                            {/* Chat Title */}
                            <div 
                              className="font-semibold text-base text-gray-900 mb-2"
                              style={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                              title={currentChat.title}
                            >
                              {currentChat.title}
                            </div>
                            
                            {/* Preview text */}
                            <div className="text-sm text-gray-600 mb-3">
                              {(() => {
                                const lastMessage = currentChat.messages && currentChat.messages.length > 0 
                                  ? currentChat.messages[currentChat.messages.length - 1] 
                                  : null
                                const preview = lastMessage 
                                  ? (lastMessage.type === 'user' ? lastMessage.content : 'AI Analysis')
                                  : 'New Session'
                                return truncateText(preview, 80)
                              })()}
                            </div>
                            
                            {/* Metadata */}
                            <div className="flex items-center justify-between text-xs text-gray-500">
                              <div className="flex items-center gap-1">
                                <Clock className="h-3 w-3 flex-shrink-0" />
                                <span>Updated {formatTime(currentChat.updated_at)}</span>
                              </div>
                              {currentChat.messages && currentChat.messages.length > 0 && (
                                <div className="flex items-center gap-1">
                                  <MessageSquare className="h-3 w-3" />
                                  <span>{currentChat.messages.length} messages</span>
                                </div>
                              )}
                            </div>
                          </div>
                          
                          {/* Actions Button */}
                          <div className="relative flex justify-center">
                            <button
                              onClick={toggleDropdown}
                              className={cn(
                                "w-8 h-8 rounded-md flex items-center justify-center",
                                "text-gray-500 hover:text-gray-700 hover:bg-gray-200",
                                "transition-colors duration-200",
                                openDropdownId === currentChatId && "bg-gray-200 text-gray-700"
                              )}
                              title="Session options"
                            >
                              <MoreVertical className="h-4 w-4" />
                            </button>
                            
                            {/* Dropdown Menu */}
                            {openDropdownId === currentChatId && (
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
                                  onClick={handleRenameClick}
                                  className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3 transition-colors"
                                >
                                  <Edit2 className="h-4 w-4 text-blue-500" />
                                  <span>Rename</span>
                                </button>
                                
                                {/* Delete Option */}
                                <button
                                  onClick={handleDeleteClick}
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
                  )}

                  {/* Quick Access to Recent Sessions */}
                  {chats.length > 0 && (
                    <div className="mt-4">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setRecentChatsModalOpen(true)}
                        className="w-full text-sm justify-center"
                      >
                        <History className="h-4 w-4 mr-2" />
                        View All {chats.length} Sessions
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
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

      {/* Recent Chats Modal */}
      <RecentChatsModal
        open={recentChatsModalOpen}
        onOpenChange={setRecentChatsModalOpen}
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={switchToChat}
        onDeleteChat={handleDeleteFromModal}
        onRenameChat={handleRenameFromModal}
      />
    </>
  )
}