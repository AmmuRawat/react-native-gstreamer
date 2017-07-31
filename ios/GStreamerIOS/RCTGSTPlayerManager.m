//
//  RNTMapManager.m
//  GStreamerIOS
//
//  Created by Alann Sapone on 21/07/2017.
//  Copyright © 2017 Facebook. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <React/RCTViewManager.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTLog.h>

#import "RCTGSTPlayerManager.h"
#import "RCTGSTPlayerController.h"

@implementation RCTGSTPlayerManager

RCT_EXPORT_MODULE();
RCT_EXPORT_VIEW_PROPERTY(defaultUri, NSString);

@synthesize bridge = _bridge;

- (UIView *)view
{
  RCTGSTPlayerController *rctGstPlayer = [[UIStoryboard storyboardWithName:@"Storyboard" bundle:nil] instantiateViewControllerWithIdentifier:@"RCTGSTPlayerController"];

  return rctGstPlayer.view;
}
@end
