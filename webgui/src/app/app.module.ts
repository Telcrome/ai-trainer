import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { LogComponent } from './logging/log/log.component';
import { RouterModule } from '@angular/router';
import { HomeComponent } from './main/home/home.component';
import { CellComponent } from './logging/cell/cell.component';
import { SocketioService } from './socketio.service';



@NgModule({
  declarations: [
    AppComponent,
    LogComponent,
    HomeComponent,
    CellComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    RouterModule.forRoot([
      { path: '', component: HomeComponent },
      { path: 'debug', component: LogComponent },
    ]),
    FormsModule,
    HttpClientModule
  ],
  providers: [SocketioService],
  bootstrap: [AppComponent]
})
export class AppModule { }
