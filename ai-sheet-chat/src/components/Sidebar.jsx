'use client'

import { useState, useCallback } from 'react'
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
  Sparkles,
  TrendingUp,
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
import AIInsightsModal from '@/components/AIInsightsModal'
import RenameChatDialog from '@/components/RenameChatDialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export default function Sidebar() {
  const {
    currentChatId,
    datasetLoaded,
    chats,
    datasetInfo,
    insights,
    progress,
    isUploading,
    uploadError,
    createNewChat,
    switchToChat,
    deleteChat,
    renameChat,
    uploadFile
  } = useAISheetChat()

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [chatToDelete, setChatToDelete] = useState(null)
  const [insightsModalOpen, setInsightsModalOpen] = useState(false)
  const [renameDialogOpen, setRenameDialogOpen] = useState(false)
  const [chatToRename, setChatToRename] = useState(null)

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
    disabled: isUploading,
    maxSize: 10 * 1024 * 1024 * 1024,
  })

  const handleDeleteClick = (e, chat) => {
    e.stopPropagation()
    setChatToDelete(chat)
    setDeleteDialogOpen(true)
  }

  const handleDeleteConfirm = () => {
    if (chatToDelete) {
      deleteChat(chatToDelete.id)
      setChatToDelete(null)
    }
  }

  const handleRenameClick = (e, chat) => {
    e.stopPropagation()
    setChatToRename(chat)
    setRenameDialogOpen(true)
  }

  const handleRenameConfirm = (newTitle) => {
    if (chatToRename) {
      renameChat(chatToRename.id, newTitle)
      setChatToRename(null)
    }
  }

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

  return (
    <>
      <div className="w-80 bg-card border-r border-border flex flex-col h-full shadow-sm">
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
          </div>
          
          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all",
              "hover:border-primary hover:bg-primary/5",
              isDragActive && !isDragReject && "border-primary bg-primary/5",
              isDragReject && "border-red-300 bg-red-50",
              isUploading && "opacity-60 cursor-not-allowed",
              uploadError && "border-red-300 bg-red-50"
            )}
          >
            <input 
              {...getInputProps()} 
              onChange={handleFileInputChange}
              disabled={isUploading}
            />
            
            <div className="relative z-10">
              {isUploading ? (
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
                  <div className="text-sm font-medium text-green-700">Dataset ready</div>
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
                    <span className="text-xs font-medium text-green-800">AI Analysis Ready</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Navigation Links */}
        <div className="px-6 py-4 border-b border-border">
          <div className="space-y-2">
            {insights.length > 0 && (
              <Button
                variant="ghost"
                className="w-full justify-start p-3 h-auto hover:bg-primary/10"
                onClick={() => setInsightsModalOpen(true)}
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Sparkles className="h-4 w-4 text-purple-600" />
                  </div>
                  <div className="text-left">
                    <div className="text-sm font-medium text-gray-900">AI Insights</div>
                    <div className="text-xs text-gray-500">{insights.length} insights available</div>
                  </div>
                  <Badge variant="secondary" className="ml-auto bg-purple-100 text-purple-800">
                    {insights.length}
                  </Badge>
                </div>
              </Button>
            )}
          </div>
        </div>

        {/* Chat History - Fixed Scrolling */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="p-6 pb-4 flex-shrink-0">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-gray-600" />
              <h3 className="text-sm font-semibold text-gray-800">Recent Sessions</h3>
              <Badge variant="secondary" className="text-xs">{chats.length}</Badge>
            </div>
          </div>
          
          {/* Scrollable Chat List */}
          <div className="flex-1 min-h-0 px-6">
            <ScrollArea className="h-full custom-scrollbar">
              <div className="space-y-2 pb-6">
                {chats.length === 0 ? (
                  <div className="text-center py-8">
                    <MessageSquare className="h-8 w-8 mx-auto text-gray-300 mb-2" />
                    <div className="text-sm text-gray-500">No sessions yet</div>
                    <div className="text-xs text-gray-400">Start by uploading data</div>
                  </div>
                ) : (
                  chats.map((chat) => {
                    const lastMessage = chat.messages && chat.messages.length > 0 
                      ? chat.messages[chat.messages.length - 1] 
                      : null
                    const preview = lastMessage 
                      ? (lastMessage.type === 'user' ? lastMessage.content : 'AI Analysis')
                      : 'New Session'
                    
                    return (
                      <div
                        key={chat.id}
                        onClick={() => switchToChat(chat.id)}
                        className={cn(
                          "group relative p-3 rounded-lg cursor-pointer transition-all border",
                          "hover:shadow-md hover:bg-gray-50",
                          chat.id === currentChatId 
                            ? "bg-purple-50 border-purple-200 shadow-sm" 
                            : "bg-white border-gray-100 hover:border-gray-200"
                        )}
                      >
                        <div className="pr-8">
                          <div className="flex items-center gap-2 mb-2">
                            <div className={cn(
                              "w-2 h-2 rounded-full",
                              chat.id === currentChatId ? "bg-purple-500" : "bg-gray-300"
                            )} />
                            <div className="font-medium text-sm text-gray-900 truncate flex-1">
                              {chat.title}
                            </div>
                          </div>
                          
                          <div className="text-xs text-gray-500 mb-2 line-clamp-2">
                            {preview.substring(0, 50)}{preview.length > 50 ? '...' : ''}
                          </div>
                          
                          <div className="flex items-center justify-between text-xs text-gray-400">
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatTime(chat.updated_at)}
                            </div>
                            {chat.messages && chat.messages.length > 0 && (
                              <div className="flex items-center gap-1">
                                <MessageSquare className="h-3 w-3" />
                                <span>{chat.messages.length}</span>
                              </div>
                            )}
                          </div>
                        </div>
                        
                        {/* Chat Actions Dropdown */}
                        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-6 w-6 p-0 hover:bg-gray-200"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <MoreVertical className="h-3 w-3" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end" className="w-48">
                              <DropdownMenuItem onClick={(e) => handleRenameClick(e, chat)}>
                                <Edit2 className="h-3 w-3 mr-2" />
                                Rename
                              </DropdownMenuItem>
                              <DropdownMenuItem 
                                onClick={(e) => handleDeleteClick(e, chat)}
                                className="text-red-600 hover:text-red-700 hover:bg-red-50"
                              >
                                <Trash2 className="h-3 w-3 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
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
        onOpenChange={setDeleteDialogOpen}
        onConfirm={handleDeleteConfirm}
        chatTitle={chatToDelete?.title || ''}
      />

      {/* Rename Dialog */}
      <RenameChatDialog
        open={renameDialogOpen}
        onOpenChange={setRenameDialogOpen}
        onConfirm={handleRenameConfirm}
        currentTitle={chatToRename?.title || ''}
      />

      {/* AI Insights Modal */}
      <AIInsightsModal
        open={insightsModalOpen}
        onOpenChange={setInsightsModalOpen}
        insights={insights}
      />
    </>
  )
}