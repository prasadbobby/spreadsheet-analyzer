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
import { Card, CardContent } from "@/components/ui/card"
import { Download, FileText, Image, Loader2 } from "lucide-react"

export default function ExportSessionDialog({ open, onOpenChange, onExport }) {
  const [isExporting, setIsExporting] = useState(false)
  const [exportType, setExportType] = useState(null)

  const handleExport = async (type) => {
    setExportType(type)
    setIsExporting(true)
    
    try {
      await onExport(type)
    } catch (error) {
      console.error('Export failed:', error)
    } finally {
      setIsExporting(false)
      setExportType(null)
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <Download className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <DialogTitle className="text-left">Export Chat Session</DialogTitle>
              <DialogDescription className="text-left">
                Choose how you'd like to export your chat session
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {/* PDF Export Option */}
          <Card 
            className={`cursor-pointer transition-all hover:shadow-md border-2 ${
              exportType === 'pdf' ? 'border-blue-300 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => !isExporting && handleExport('pdf')}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                  {isExporting && exportType === 'pdf' ? (
                    <Loader2 className="h-5 w-5 text-red-600 animate-spin" />
                  ) : (
                    <FileText className="h-5 w-5 text-red-600" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">Export as PDF</h3>
                  <p className="text-sm text-gray-600">
                    Generate a formatted PDF document with all messages and data
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Image Export Option */}
          <Card 
            className={`cursor-pointer transition-all hover:shadow-md border-2 ${
              exportType === 'image' ? 'border-blue-300 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => !isExporting && handleExport('image')}
          >
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  {isExporting && exportType === 'image' ? (
                    <Loader2 className="h-5 w-5 text-green-600 animate-spin" />
                  ) : (
                    <Image className="h-5 w-5 text-green-600" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">Export as Image</h3>
                  <p className="text-sm text-gray-600">
                    Capture a full-page screenshot of the entire chat session
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isExporting}
          >
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}