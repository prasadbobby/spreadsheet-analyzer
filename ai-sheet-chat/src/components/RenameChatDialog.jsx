'use client'

import { useState } from 'react'
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
  const [newTitle, setNewTitle] = useState(currentTitle || '')

  const handleSubmit = () => {
    if (newTitle.trim() && newTitle.trim() !== currentTitle) {
      onConfirm(newTitle.trim())
      onOpenChange(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleSubmit()
    }
  }

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
          <Input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter new chat name..."
            maxLength={100}
            autoFocus
            className="w-full"
          />
          <p className="text-xs text-gray-500 mt-2">
            Maximum 100 characters
          </p>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!newTitle.trim() || newTitle.trim() === currentTitle}
            style={{ backgroundColor: '#8500df' }}
            className="text-white hover:opacity-90"
          >
            Rename
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}