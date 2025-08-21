import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

export function formatTime(timestamp) {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`

  return date.toLocaleDateString()
}

export function getCurrentTime() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  })
}

export function escapeHtml(text) {
  if (typeof text !== 'string') {
    text = String(text)
  }
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

export function formatMessageText(text) {
  if (!text) return ''
  
  // Convert markdown-style formatting
  let formatted = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code class="bg-muted px-1 py-0.5 rounded text-sm">$1</code>')
  
  // Enhanced handling for the "complete data used for this analysis" section
  formatted = formatted.replace(
    /(The complete data used (?:for this analysis|to derive this result) is available in the table below:\s*)`\s*(\[.*?\])\s*`/gi,
    (match, prefix, jsonData) => {
      try {
        const data = JSON.parse(jsonData)
        if (Array.isArray(data) && data.length > 0) {
          const firstItem = data[0]
          if (typeof firstItem === 'object') {
            return `${prefix}<div class="mt-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
              <div class="flex items-center gap-2 mb-3">
                <svg class="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"></path>
                </svg>
                <span class="text-sm font-semibold text-blue-800">Analysis Data Summary</span>
                <span class="text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full">${data.length} record${data.length > 1 ? 's' : ''}</span>
              </div>
              <div class="space-y-2">
                ${data.slice(0, 3).map(item => {
                  return `<div class="bg-white p-3 rounded-md border border-blue-100 shadow-sm">
                    <div class="grid grid-cols-1 gap-1 text-xs">
                      ${Object.entries(item).map(([key, value]) => 
                        `<div class="flex justify-between">
                          <span class="font-medium text-gray-600 capitalize">${key.replace(/_/g, ' ')}:</span>
                          <span class="text-gray-900 font-mono">${typeof value === 'number' && key.toLowerCase().includes('salary') ? 
                            '$' + value.toLocaleString() : value}</span>
                        </div>`
                      ).join('')}
                    </div>
                  </div>`
                }).join('')}
                ${data.length > 3 ? `<div class="text-xs text-blue-600 text-center mt-2">... and ${data.length - 3} more records in the table below</div>` : ''}
              </div>
            </div>`
          }
        }
      } catch (e) {
        // If parsing fails, return a clean formatted version
        return `${prefix}<div class="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div class="flex items-center gap-2 text-sm text-gray-700">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            Complete analysis data is displayed in the table below
          </div>
        </div>`
      }
      return match
    }
  )
  
  // Handle any remaining raw JSON arrays (fallback)
  formatted = formatted.replace(/`\s*(\[.*?\])\s*`/g, (match, jsonData) => {
    try {
      const data = JSON.parse(jsonData)
      if (Array.isArray(data) && data.length > 0) {
        return `<div class="inline-flex items-center gap-2 px-3 py-1 bg-blue-50 text-blue-700 rounded-lg text-sm border border-blue-200">
          <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
            <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"></path>
          </svg>
          ${data.length} record${data.length > 1 ? 's' : ''} analyzed
        </div>`
      }
    } catch (e) {
      return `<span class="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-600 rounded text-sm">
        📊 Data in table below
      </span>`
    }
    return match
  })
  
  // Convert line breaks
  formatted = formatted.replace(/\n/g, '<br>')
  
  return formatted
}

export function formatCellValue(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toLocaleString()
    return value.toFixed(2)
  }
  return String(value)
}