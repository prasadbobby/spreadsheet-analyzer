'use client'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Sparkles, TrendingUp, Lightbulb, BarChart3, Brain } from "lucide-react"

export default function AIInsightsModal({ open, onOpenChange, insights }) {
  if (!insights || insights.length === 0) {
    return null
  }

  const getInsightIcon = (index) => {
    const icons = [TrendingUp, Lightbulb, BarChart3, Brain, Sparkles]
    return icons[index % icons.length]
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <DialogTitle className="text-left text-xl">AI Insights</DialogTitle>
              <DialogDescription className="text-left">
                Auto-generated insights from your dataset analysis
              </DialogDescription>
            </div>
            <Badge variant="secondary" className="ml-auto bg-purple-100 text-purple-800">
              {insights.length} insights
            </Badge>
          </div>
        </DialogHeader>
        
        <ScrollArea className="max-h-[60vh] pr-4">
          <div className="space-y-4 mt-4">
            {insights.map((insight, index) => {
              const IconComponent = getInsightIcon(index)
              return (
                <div 
                  key={index}
                  className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                      <IconComponent className="h-4 w-4 text-purple-600" />
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-900 mb-1">
                        Insight #{index + 1}
                      </div>
                      <div className="text-sm text-gray-700 leading-relaxed">
                        {insight}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}