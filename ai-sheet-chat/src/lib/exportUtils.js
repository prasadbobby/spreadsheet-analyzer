import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'

export const exportUtils = {
  async exportAsImage(elementId = 'chat-content') {
    try {
      // Find the main content area
      const element = document.getElementById(elementId) || document.querySelector('.main-content-area')
      
      if (!element) {
        throw new Error('Chat content element not found')
      }

      // Hide scrollbars temporarily
      const originalOverflow = element.style.overflow
      element.style.overflow = 'visible'

      // Get the full height of the content
      const originalHeight = element.style.height
      element.style.height = 'auto'

      // Wait a bit for layout to settle
      await new Promise(resolve => setTimeout(resolve, 100))

      // Capture the screenshot
      const canvas = await html2canvas(element, {
        height: element.scrollHeight,
        width: element.scrollWidth,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#f9fafb',
        scale: 2, // Higher quality
        logging: false,
        onclone: (clonedDoc) => {
          // Ensure all content is visible in the clone
          const clonedElement = clonedDoc.getElementById(elementId) || clonedDoc.querySelector('.main-content-area')
          if (clonedElement) {
            clonedElement.style.height = 'auto'
            clonedElement.style.overflow = 'visible'
            
            // Remove any height constraints from scrollable areas
            const scrollAreas = clonedElement.querySelectorAll('[data-scroll-area], .scroll-area, .overflow-hidden, .overflow-auto')
            scrollAreas.forEach(area => {
              area.style.height = 'auto'
              area.style.maxHeight = 'none'
              area.style.overflow = 'visible'
            })
          }
        }
      })

      // Restore original styles
      element.style.overflow = originalOverflow
      element.style.height = originalHeight

      // Download the image
      const link = document.createElement('a')
      link.download = `chat-session-${new Date().toISOString().slice(0, 10)}.png`
      link.href = canvas.toDataURL('image/png')
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      return true
    } catch (error) {
      console.error('Error exporting as image:', error)
      throw error
    }
  },

  async exportAsPDF(elementId = 'chat-content') {
    try {
      // Find the main content area
      const element = document.getElementById(elementId) || document.querySelector('.main-content-area')
      
      if (!element) {
        throw new Error('Chat content element not found')
      }

      // Hide scrollbars temporarily
      const originalOverflow = element.style.overflow
      element.style.overflow = 'visible'

      // Get the full height of the content
      const originalHeight = element.style.height
      element.style.height = 'auto'

      // Wait a bit for layout to settle
      await new Promise(resolve => setTimeout(resolve, 100))

      // Capture the screenshot
      const canvas = await html2canvas(element, {
        height: element.scrollHeight,
        width: element.scrollWidth,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        scale: 1.5, // Good quality for PDF
        logging: false,
        onclone: (clonedDoc) => {
          // Ensure all content is visible in the clone
          const clonedElement = clonedDoc.getElementById(elementId) || clonedDoc.querySelector('.main-content-area')
          if (clonedElement) {
            clonedElement.style.height = 'auto'
            clonedElement.style.overflow = 'visible'
            
            // Remove any height constraints from scrollable areas
            const scrollAreas = clonedElement.querySelectorAll('[data-scroll-area], .scroll-area, .overflow-hidden, .overflow-auto')
            scrollAreas.forEach(area => {
              area.style.height = 'auto'
              area.style.maxHeight = 'none'
              area.style.overflow = 'visible'
            })
          }
        }
      })

      // Restore original styles
      element.style.overflow = originalOverflow
      element.style.height = originalHeight

      // Create PDF
      const imgData = canvas.toDataURL('image/png')
      const pdf = new jsPDF({
        orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
        unit: 'px',
        format: [canvas.width, canvas.height]
      })

      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height)
      
      // Download the PDF
      pdf.save(`chat-session-${new Date().toISOString().slice(0, 10)}.pdf`)

      return true
    } catch (error) {
      console.error('Error exporting as PDF:', error)
      throw error
    }
  }
}