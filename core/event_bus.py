from collections import defaultdict
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

class EventBus:
    """
    Simple event system for loose component coupling
    
    This component:
    1. Allows components to subscribe to events
    2. Enables publishing events to all subscribers
    3. Provides logging and error handling for event processing
    4. Maintains clean interfaces between components
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_counts: Dict[str, int] = defaultdict(int)
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe a callback to an event type
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed new callback to event '{event_type}'")
        
    def publish(self, event_type: str, data: Any = None) -> int:
        """
        Publish an event to all subscribers
        
        Args:
            event_type: Type of event to publish
            data: Data to pass to subscribers
            
        Returns:
            Number of subscribers notified
        """
        if event_type not in self.subscribers or not self.subscribers[event_type]:
            logger.debug(f"No subscribers for event '{event_type}'")
            return 0
            
        self.event_counts[event_type] += 1
        logger.debug(f"Publishing event '{event_type}' to {len(self.subscribers[event_type])} subscribers")
        
        success_count = 0
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
                success_count += 1
            except Exception as e:
                logger.error(f"Error in subscriber callback for event '{event_type}': {str(e)}")
                
        return success_count
        
    def get_event_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about events"""
        return {
            "events": {
                event_type: {
                    "subscriber_count": len(subscribers),
                    "publish_count": self.event_counts[event_type]
                }
                for event_type, subscribers in self.subscribers.items()
            }
        }
