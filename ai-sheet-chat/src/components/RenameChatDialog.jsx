'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Edit2 } from "lucide-react"

export default function RenameChatDialog({ open, onOpenChange, onConfirm, currentTitle }) {
  const [newTitle, setNewTitle] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const inputRef = useRef(null)

  // Initialize and reset title based on dialog state and currentTitle
  useEffect(() => {
    if (open && currentTitle) {
      setNewTitle(currentTitle)
      // Focus and select all text after a brief delay to ensure the dialog is fully rendered
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus()
          inputRef.current.select()
        }
      }, 100)
    } else if (!open) {
      // Reset state when dialog closes
      setNewTitle('')
      setIsSubmitting(false)
    }
  }, [open, currentTitle])

  const handleSubmit = async () => {
    const trimmedTitle = newTitle.trim()
    
    if (!trimmedTitle) {
      return
    }

    if (trimmedTitle === currentTitle) {
      // No change, just close
      onOpenChange(false)
      return
    }

    setIsSubmitting(true)
    
    try {
      await onConfirm(trimmedTitle)
      onOpenChange(false)
    } catch (error) {
      console.error('Error renaming chat:', error)
      // Keep dialog open on error
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      onOpenChange(false)
    }
  }

  const handleCancel = () => {
    onOpenChange(false)
  }

  const isValid = newTitle.trim().length > 0
  const hasChanged = newTitle.trim() !== currentTitle
  const canSubmit = isValid && hasChanged && !isSubmitting

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <Edit2 className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <DialogTitle className="text-left">Rename Chat</DialogTitle>
              <DialogDescription className="text-left">
                Give this chat session a new name.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        
        <div className="py-4">
          <div className="space-y-2">
            <label htmlFor="chat-name" className="text-sm font-medium text-gray-700">
              Chat Name
            </label>
            <Input
              id="chat-name"
              ref={inputRef}
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter new chat name..."
              maxLength={100}
              disabled={isSubmitting}
              className="w-full"
            />
            <div className="flex justify-between items-center">
              <p className="text-xs text-gray-500">
                Maximum 100 characters
              </p>
              <p className="text-xs text-gray-400">
                {newTitle.length}/100
              </p>
            </div>
            {newTitle.trim() && !hasChanged && (
              <p className="text-xs text-amber-600">
                Name hasn't changed
              </p>
            )}
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!canSubmit}
            style={{ backgroundColor: '#8500df' }}
            className="text-white hover:opacity-90 disabled:opacity-50"
          >
            {isSubmitting ? 'Renaming...' : 'Rename'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}