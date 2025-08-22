'use client'

import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { 
  MessageSquare, 
  Clock, 
  Search, 
  X, 
  Calendar,
  Filter,
  SortAsc,
  SortDesc,
  MoreVertical,
  Edit2,
  Trash2,
  Star,
  Archive
} from "lucide-react"
import { formatTime } from "@/lib/utils"
import { cn } from "@/lib/utils"

export default function RecentChatsModal({ 
  open, 
  onOpenChange, 
  chats = [], 
  currentChatId,
  onSelectChat,
  onDeleteChat,
  onRenameChat 
}) {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState('updated_at') // 'updated_at', 'created_at', 'title', 'message_count'
  const [sortOrder, setSortOrder] = useState('desc') // 'asc', 'desc'
  const [filterBy, setFilterBy] = useState('all') // 'all', 'today', 'week', 'month'
  const [selectedChatForActions, setSelectedChatForActions] = useState(null)

  // Filter and sort chats
  const filteredAndSortedChats = chats
    .filter(chat => {
      // Text search filter
      const matchesSearch = !searchTerm || 
        chat.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (chat.messages && chat.messages.some(msg => 
          msg.content.toLowerCase().includes(searchTerm.toLowerCase())
        ))

      if (!matchesSearch) return false

      // Date filter
      if (filterBy === 'all') return true

      const chatDate = new Date(chat.updated_at)
      const now = new Date()
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)
      const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000)

      switch (filterBy) {
        case 'today':
          return chatDate >= today
        case 'week':
          return chatDate >= weekAgo
        case 'month':
          return chatDate >= monthAgo
        default:
          return true
      }
    })
    .sort((a, b) => {
      let aValue, bValue

      switch (sortBy) {
        case 'title':
          aValue = a.title.toLowerCase()
          bValue = b.title.toLowerCase()
          break
        case 'created_at':
          aValue = new Date(a.created_at)
          bValue = new Date(b.created_at)
          break
        case 'message_count':
          aValue = a.messages ? a.messages.length : 0
          bValue = b.messages ? b.messages.length : 0
          break
        case 'updated_at':
        default:
          aValue = new Date(a.updated_at)
          bValue = new Date(b.updated_at)
          break
      }

      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })

  const handleSelectChat = (chat) => {
    onSelectChat(chat.id)
    onOpenChange(false)
  }

  const toggleSort = (newSortBy) => {
    if (sortBy === newSortBy) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(newSortBy)
      setSortOrder('desc')
    }
  }

  const getLastMessage = (chat) => {
    if (!chat.messages || chat.messages.length === 0) {
      return 'No messages yet'
    }
    const lastMessage = chat.messages[chat.messages.length - 1]
    const content = lastMessage.content.length > 60 
      ? lastMessage.content.substring(0, 60) + '...'
      : lastMessage.content
    return lastMessage.type === 'user' ? `You: ${content}` : `AI: ${content}`
  }

  const getChatStats = () => {
    const total = chats.length
    const today = chats.filter(chat => {
      const chatDate = new Date(chat.updated_at)
      const todayDate = new Date()
      return chatDate.toDateString() === todayDate.toDateString()
    }).length
    
    const thisWeek = chats.filter(chat => {
      const chatDate = new Date(chat.updated_at)
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
      return chatDate >= weekAgo
    }).length

    return { total, today, thisWeek }
  }

  const stats = getChatStats()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                <MessageSquare className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <DialogTitle className="text-left text-xl">Recent Chat Sessions</DialogTitle>
                <p className="text-sm text-gray-600 text-left">
                  Manage and browse your analysis sessions
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                {stats.total} total
              </Badge>
              {stats.today > 0 && (
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  {stats.today} today
                </Badge>
              )}
            </div>
          </div>
        </DialogHeader>

        {/* Search and Filters */}
        <div className="space-y-4 border-b pb-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search chats by title or content..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-10"
            />
            {searchTerm && (
              <Button
                size="sm"
                variant="ghost"
                className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
                onClick={() => setSearchTerm('')}
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>

          {/* Filters and Sort */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <select
                value={filterBy}
                onChange={(e) => setFilterBy(e.target.value)}
                className="text-sm border border-gray-200 rounded px-2 py-1"
              >
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">This Week</option>
                <option value="month">This Month</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500">Sort by:</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleSort('updated_at')}
                className={cn(
                  "h-8",
                  sortBy === 'updated_at' && "bg-gray-100"
                )}
              >
                Last Updated
                {sortBy === 'updated_at' && (
                  sortOrder === 'desc' ? <SortDesc className="h-3 w-3 ml-1" /> : <SortAsc className="h-3 w-3 ml-1" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleSort('title')}
                className={cn(
                  "h-8",
                  sortBy === 'title' && "bg-gray-100"
                )}
              >
                Name
                {sortBy === 'title' && (
                  sortOrder === 'desc' ? <SortDesc className="h-3 w-3 ml-1" /> : <SortAsc className="h-3 w-3 ml-1" />
                )}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleSort('message_count')}
                className={cn(
                  "h-8",
                  sortBy === 'message_count' && "bg-gray-100"
                )}
              >
                Messages
                {sortBy === 'message_count' && (
                  sortOrder === 'desc' ? <SortDesc className="h-3 w-3 ml-1" /> : <SortAsc className="h-3 w-3 ml-1" />
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Chat List */}
        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full">
            {filteredAndSortedChats.length === 0 ? (
              <div className="text-center py-12">
                <MessageSquare className="h-12 w-12 mx-auto text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {searchTerm || filterBy !== 'all' ? 'No matching chats found' : 'No chat sessions yet'}
                </h3>
                <p className="text-gray-500 max-w-md mx-auto">
                  {searchTerm || filterBy !== 'all' 
                    ? 'Try adjusting your search terms or filters to find what you\'re looking for.'
                    : 'Create your first analysis session to start chatting with your data.'
                  }
                </p>
              </div>
            ) : (
              <div className="space-y-2 p-1">
                {filteredAndSortedChats.map((chat) => (
                  <div
                    key={chat.id}
                    className={cn(
                      "group relative rounded-lg border transition-all cursor-pointer",
                      "hover:shadow-md hover:bg-gray-50",
                      chat.id === currentChatId 
                        ? "bg-purple-50 border-purple-200 shadow-sm" 
                        : "bg-white border-gray-100 hover:border-gray-200"
                    )}
                    onClick={() => handleSelectChat(chat)}
                  >
                    <div className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          {/* Chat Title and Status */}
                          <div className="flex items-center gap-2 mb-2">
                            <div className={cn(
                              "w-2 h-2 rounded-full flex-shrink-0",
                              chat.id === currentChatId ? "bg-purple-500" : "bg-gray-300"
                            )} />
                            <h3 className="font-medium text-gray-900 truncate flex-1">
                              {chat.title}
                            </h3>
                            {chat.messages && chat.messages.length > 0 && (
                              <Badge variant="secondary" className="text-xs">
                                {chat.messages.length}
                              </Badge>
                            )}
                          </div>

                          {/* Last Message Preview */}
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                            {getLastMessage(chat)}
                          </p>

                          {/* Metadata */}
                          <div className="flex items-center justify-between text-xs text-gray-500">
                            <div className="flex items-center gap-4">
                              <div className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                <span>Updated {formatTime(chat.updated_at)}</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                <span>Created {formatTime(chat.created_at)}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Actions Menu */}
                        <div className="relative ml-4">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="opacity-0 group-hover:opacity-100 h-8 w-8 p-0"
                            onClick={(e) => {
                              e.stopPropagation()
                              setSelectedChatForActions(
                                selectedChatForActions === chat.id ? null : chat.id
                              )
                            }}
                          >
                            <MoreVertical className="h-4 w-4" />
                          </Button>

                          {/* Dropdown Menu */}
                          {selectedChatForActions === chat.id && (
                            <div 
                              className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-[140px] z-50"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  onRenameChat(chat.id, chat.title)
                                  setSelectedChatForActions(null)
                                }}
                                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-3"
                              >
                                <Edit2 className="h-4 w-4" />
                                Rename
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  onDeleteChat(chat.id, chat.title)
                                  setSelectedChatForActions(null)
                                }}
                                className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-3"
                              >
                                <Trash2 className="h-4 w-4" />
                                Delete
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Footer with Statistics */}
        <div className="border-t pt-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span>Showing {filteredAndSortedChats.length} of {chats.length} sessions</span>
              {filterBy !== 'all' && (
                <Badge variant="outline" className="text-xs">
                  Filtered by {filterBy}
                </Badge>
              )}
            </div>
            <Button onClick={() => onOpenChange(false)}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}